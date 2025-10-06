from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd


@dataclass
class CSVToolResult:
    rows: int
    preview: str


class CSVTool:
    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)

    def run(self, filename: str, query: Optional[str] = None) -> CSVToolResult:
        path = (self.data_dir / filename).resolve()
        if not str(path).startswith(str(self.data_dir.resolve())):
            raise ValueError("CSVTool path traversal detected")
        if not path.exists():
            raise FileNotFoundError(str(path))
        df = pd.read_csv(path)
        if query:
            df = df.query(query)
        preview = df.head(10).to_csv(index=False)
        return CSVToolResult(rows=len(df), preview=preview)
