import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from services.excel_loader import ExcelLoader
from services.rag_service import RAGService, clean_text, summarize_text


def test_loader_reads_csv_with_alias_columns(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "Problem Description": "Material received but PO pending",
                "Effort / Action Taken": "Called supplier and reopened order",
                "Level": "L2",
                "Status": "Open",
                "Solution": "Update the PO entry in the system",
            }
        ]
    ).to_csv(csv_path, index=False)

    loader = ExcelLoader(csv_path)
    df = loader.load()

    assert list(df.columns) == ["cause", "prevention", "effort_taken", "status", "solution", "level"]
    assert df.iloc[0]["cause"] == "Material received but PO pending"
    # effort_taken holds the REAL actions taken (not the escalation level).
    assert df.iloc[0]["effort_taken"] == "Called supplier and reopened order"
    # prevention holds the final resolution / fix.
    assert df.iloc[0]["prevention"] == "Update the PO entry in the system"
    assert df.iloc[0]["status"] == "Open"
    assert df.iloc[0]["level"] == "L2"


def test_loader_merges_multi_level_rows_into_one_record(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "Escalation ID": "ESC-002",
                "Level": "L1",
                "Problem Description": "Coating area thki as per Drawing size",
                "Effort / Action Taken": "Suggestions to mail to customer",
                "Status": "Open",
                "Timestamp": "4/18/2026 14:06:47",
                "Solution": "",
            },
            {
                "Escalation ID": "ESC-002",
                "Level": "L2",
                "Problem Description": "Coating area thki as per Drawing size",
                "Effort / Action Taken": "Must inform customer via mail and confirm",
                "Status": "Close",
                "Timestamp": "4/18/2026 14:28:08",
                "Solution": "Change the code in PIR. 0.200 mm required as per drawing.",
            },
        ]
    ).to_csv(csv_path, index=False)

    df = ExcelLoader(csv_path).load()

    # Both L1 and L2 rows collapse into ONE record.
    assert len(df) == 1
    row = df.iloc[0]
    assert row["level"] == "L1 → L2"
    # The full effort trail from BOTH levels is preserved.
    assert "Suggestions to mail to customer" in row["effort_taken"]
    assert "Must inform customer via mail and confirm" in row["effort_taken"]
    # The solution recorded at L2 is NOT lost (previously it was).
    assert row["prevention"] == "Change the code in PIR. 0.200 mm required as per drawing."
    assert row["solution"] == row["prevention"]
    # Status reflects the FINAL state and timestamp the LATEST update.
    assert row["status"] == "Close"
    assert row["timestamp"] == "4/18/2026 14:28:08"


def test_clean_text_improves_handover_phrase_and_sentence_case() -> None:
    assert clean_text("the customer could not handover the material") == "The customer could not hand over the material"


def test_summarize_text_keeps_the_first_sentence_short() -> None:
    text = "The customer reported a shipping delay. We contacted the supplier and reopened the order."
    assert summarize_text(text) == "The customer reported a shipping delay."


def test_rag_falls_back_to_generalized_summary_for_new_cause(tmp_path: Path) -> None:
    csv_path = tmp_path / "sample.csv"
    pd.DataFrame(
        [
            {
                "Problem Description": "Material received but PO pending",
                "Effort / Action Taken": "Called supplier and reopened order",
                "Level": "L2",
                "Status": "Open",
                "Escalation ID": "E-100",
                "Timestamp": "2024-01-01",
                "Department": "Supply Chain",
                "Solution": "Reopen the PO and confirm with the supplier",
            }
        ]
    ).to_csv(csv_path, index=False)

    service = RAGService(csv_path)
    result = service.search("totally new issue about shipment delay")

    assert result["recommended_solution"]
    assert result["recommended_solution"] != "No similar historical escalation found."
    assert result["similar_cases"]
