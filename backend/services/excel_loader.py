from __future__ import annotations

import os
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd

from services.text_cleaning import clean_text


class ExcelLoader:
    """Load escalation records from Excel/CSV with accurate normalization.

    The raw escalation log stores one row per escalation LEVEL (L1, L2, ...)
    for the same escalation. Those rows are merged into ONE record per
    escalation so that no information is lost:

    - cause        : first non-empty problem description (cleaned)
    - effort_taken : every distinct non-empty "Effort / Action Taken" entry,
                     joined with " | " (the full escalation trail)
    - prevention   : first non-empty Solution (the final resolution/fix)
    - solution     : same as prevention (backward-compatible alias)
    - status       : last non-empty Status (the final state)
    - timestamp    : most recent parseable Timestamp (kept verbatim)
    - level        : distinct levels joined, e.g. "L1 → L2"
    - metadata     : escalation_id, department, kp_no, customer_name,
                     material_name, raised_by (first non-empty, verbatim)
    """

    #: Output column order (only fields mapped from the source file are kept).
    OUTPUT_ORDER = [
        "cause",
        "prevention",
        "effort_taken",
        "status",
        "escalation_id",
        "timestamp",
        "department",
        "solution",
        "kp_no",
        "customer_name",
        "material_name",
        "raised_by",
        "level",
    ]

    #: Mapping of normalized field -> source column aliases (first match wins).
    COLUMN_ALIASES = {
        "cause": ["Cause", "Problem Description", "Problem Description ", "Description"],
        "prevention": ["Prevention", "Solution"],
        "effort_taken": ["Effort Taken", "Effort / Action Taken", "Effort / Action Taken ", "Action Taken"],
        "status": ["Status", "status"],
        "escalation_id": ["Escalation ID", "Escalation ID ", "Escalation Id"],
        "timestamp": ["Timestamp", "Raised On", "Time", "Created On"],
        "department": ["Department", "Department Name", "Dept"],
        "solution": ["Solution"],
        "kp_no": ["KP No.", "KP No", "KP Number"],
        "customer_name": ["Customer Name", "Customer"],
        "material_name": ["Material Name", "Material"],
        "raised_by": ["Raised By"],
        "level": ["Level", "Level Raised"],
    }

    def __init__(self, data_path: str | os.PathLike[str]) -> None:
        self.data_path = Path(data_path)
        # normalized field -> actual source header, captured on load()
        self._source_headers: dict[str, str] = {}


    #: Fields that must exist in the source file (values may be empty).
    REQUIRED_FIELDS = ["cause", "prevention", "effort_taken", "status"]

    #: Fields preserved verbatim (no whitespace normalization / no cleaning).
    PRESERVE_EXACT_FIELDS = {
        "escalation_id",
        "department",
        "timestamp",
        "solution",
        "kp_no",
        "customer_name",
        "material_name",
        "raised_by",
        "level",
    }

    #: Canonical source headers used when creating a brand-new data file.
    CANONICAL_HEADERS = {
        "cause": "Problem Description",
        "prevention": "Solution",
        "effort_taken": "Effort / Action Taken",
        "status": "Status",
        "escalation_id": "Escalation ID",
        "timestamp": "Timestamp",
        "department": "Department",
        "solution": "Solution",
        "kp_no": "KP No.",
        "customer_name": "Customer Name",
        "material_name": "Material Name",
        "raised_by": "Raised By",
        "level": "Level",
    }

    def _read_dataframe(self) -> pd.DataFrame:
        if self.data_path.suffix.lower() == ".csv":
            return pd.read_csv(self.data_path, dtype=str, keep_default_na=False)
        return pd.read_excel(self.data_path, engine="openpyxl", dtype=str, keep_default_na=False)

    def load(self) -> pd.DataFrame:
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")

        raw = self._read_dataframe()
        raw.columns = [str(column).strip() for column in raw.columns]

        # Resolve normalized field -> actual source column (first alias match).
        source_of: dict[str, str] = {}
        for target, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in raw.columns:
                    source_of[target] = alias
                    break
        self._source_headers = dict(source_of)

        missing = [field for field in self.REQUIRED_FIELDS if field not in source_of]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")

        # Normalized working copy (strings only, no NaN).
        norm = pd.DataFrame(index=raw.index)
        for target, source in source_of.items():
            column = raw[source].astype(str).fillna("")
            if target not in self.PRESERVE_EXACT_FIELDS:
                column = column.str.strip()
            norm[target] = column

        records = [self._merge_group(group) for _, group in self._grouped(norm)]

        if not records:
            return pd.DataFrame(columns=[column for column in self.OUTPUT_ORDER])

        df = pd.DataFrame(records)
        df = df[df["cause"].astype(str).str.strip() != ""].copy()
        df = df.drop_duplicates(subset=["cause"], keep="first")
        ordered = [column for column in self.OUTPUT_ORDER if column in df.columns]
        return df[ordered].reset_index(drop=True)

    @staticmethod
    def _grouped(norm: pd.DataFrame):
        """Group rows belonging to the same escalation (by escalation id,
        falling back to the normalized cause text when the id is missing)."""
        keys: list[str] = []
        for row_index in norm.index:
            escalation_id = str(norm.at[row_index, "escalation_id"]).strip() if "escalation_id" in norm.columns else ""
            cause = str(norm.at[row_index, "cause"]).strip().lower()
            keys.append(escalation_id if escalation_id else f"cause::{cause}")
        work = norm.copy()
        work["_group"] = keys
        return work.groupby("_group", sort=False)

    # ------------------------------------------------------------------
    # Merging multi-level rows into one record per escalation
    # ------------------------------------------------------------------
    def _merge_group(self, group: pd.DataFrame) -> dict[str, Any]:
        record: dict[str, Any] = {}

        # Cause: first non-empty, cleaned the same way as search queries.
        record["cause"] = clean_text(self._first_non_empty(group["cause"]))

        # Effort trail: every distinct non-empty action across all levels.
        efforts: list[str] = []
        if "effort_taken" in group.columns:
            for value in group["effort_taken"]:
                text = clean_text(value)
                if text and text not in efforts:
                    efforts.append(text)
        record["effort_taken"] = " | ".join(efforts)

        # Prevention / solution: first non-empty Solution = final resolution.
        resolution = ""
        if "solution" in group.columns:
            resolution = self._first_non_empty(group["solution"])
        elif "prevention" in group.columns:
            resolution = self._first_non_empty(group["prevention"])
        record["prevention"] = resolution
        record["solution"] = resolution  # backward-compatible alias

        # Status: last non-empty entry (final state of the escalation).
        record["status"] = self._last_non_empty(group["status"]) if "status" in group.columns else ""

        # Timestamp: most recent parseable timestamp, kept verbatim.
        if "timestamp" in group.columns:
            record["timestamp"] = self._latest_timestamp(group["timestamp"])

        # Verbatim metadata: first non-empty value per field.
        for field in ("escalation_id", "department", "kp_no", "customer_name", "material_name", "raised_by"):
            if field in group.columns:
                record[field] = self._first_non_empty(group[field])

        # Level trail: distinct levels in file order, e.g. "L1 → L2".
        if "level" in group.columns:
            levels = list(dict.fromkeys(str(value).strip() for value in group["level"] if str(value).strip()))
            record["level"] = " → ".join(levels)

        return record

    @staticmethod
    def _first_non_empty(column: pd.Series) -> str:
        for value in column:
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        return ""

    @staticmethod
    def _last_non_empty(column: pd.Series) -> str:
        for value in reversed(list(column)):
            text = str(value).strip()
            if text and text.lower() != "nan":
                return text
        return ""

    @staticmethod
    def _latest_timestamp(column: pd.Series) -> str:
        parsed = pd.to_datetime(column, errors="coerce", dayfirst=False)
        if parsed.notna().any():
            return str(column.loc[parsed.idxmax()]).strip()
        return ExcelLoader._first_non_empty(column)

    # ------------------------------------------------------------------
    # Writing (always in the ORIGINAL file schema)
    # ------------------------------------------------------------------
    def _resolve_write_headers(self) -> list[str]:
        """Return the source file's original headers (or canonical ones for a
        new file) so written data never changes the file's schema."""
        if self.data_path.exists():
            existing = pd.read_csv(self.data_path, dtype=str, keep_default_na=False, nrows=0)
            return [str(column).strip() for column in existing.columns]
        ordered: list[str] = []
        for target in self.OUTPUT_ORDER:
            header = self.CANONICAL_HEADERS.get(target)
            if header and header not in ordered:
                ordered.append(header)
        return ordered

    def _header_for(self, target: str, headers: list[str]) -> str | None:
        """Pick the file header that corresponds to a normalized field."""
        captured = self._source_headers.get(target)
        if captured and captured in headers:
            return captured
        canonical = self.CANONICAL_HEADERS.get(target)
        if canonical and canonical in headers:
            return canonical
        return None

    def append_record(self, record: dict[str, Any]) -> str:
        """Append one new escalation to the source file, preserving the
        original rows, levels and column headers. Returns the assigned
        escalation id."""
        headers = self._resolve_write_headers()

        if self.data_path.exists():
            existing = pd.read_csv(self.data_path, dtype=str, keep_default_na=False)
            existing.columns = [str(column).strip() for column in existing.columns]
        else:
            existing = pd.DataFrame(columns=headers)

        # Assign the next escalation id when none was provided.
        escalation_id = str(record.get("escalation_id", "") or "").strip()
        if not escalation_id and "Escalation ID" in existing.columns:
            numbers = []
            for value in existing["Escalation ID"].astype(str):
                match = re.search(r"(\d+)\s*$", value.strip())
                if match:
                    numbers.append(int(match.group(1)))
            escalation_id = f"ESC-{max(numbers) + 1:03d}" if numbers else "ESC-001"

        new_row: dict[str, Any] = {header: "" for header in headers}
        for target, value in record.items():
            header = self._header_for(target, headers)
            if header is None:
                continue
            if target == "prevention" and "Solution" in headers:
                header = "Solution"  # prevention and solution share one column
            new_row[header] = str(value)

        if "Timestamp" in new_row and not str(new_row.get("Timestamp", "")).strip():
            new_row["Timestamp"] = datetime.now().strftime("%m/%d/%Y %H:%M:%S")
        if "Escalation ID" in new_row:
            new_row["Escalation ID"] = escalation_id

        updated = pd.concat([existing, pd.DataFrame([new_row])], ignore_index=True)
        self._write(updated[list(existing.columns)])
        return escalation_id

    def save(self, records: pd.DataFrame) -> None:
        """Rewrite the whole data file in its ORIGINAL schema.

        Normalized column names are mapped back to the source file's original
        headers so the file schema never drifts.
        """
        headers = self._resolve_write_headers()
        out = pd.DataFrame({header: [""] * len(records) for header in headers})
        used_headers: set[str] = set()
        for target in records.columns:
            header = self._header_for(target, headers)
            if header is None or header in used_headers:
                continue
            out[header] = records[target].astype(str).tolist()
            used_headers.add(header)
        self._write(out[headers])

    def _write(self, dataframe: pd.DataFrame) -> None:
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        if self.data_path.suffix.lower() == ".csv":
            dataframe.to_csv(self.data_path, index=False)
            return
        dataframe.to_excel(self.data_path, index=False, engine="openpyxl")
