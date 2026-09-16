from __future__ import annotations

from services.excel_loader import ExcelLoader


class SaveService:
    def __init__(self, loader: ExcelLoader) -> None:
        self.loader = loader
