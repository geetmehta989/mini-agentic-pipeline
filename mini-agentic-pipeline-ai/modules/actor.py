import os
from dataclasses import dataclass
from typing import Any, Dict, Optional, List

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore
import csv

@dataclass
class ToolResult:
    tool: str
    input: Dict[str, Any]
    output: Any

class Actor:
    def __init__(self, tools_dir: str):
        self.tools_dir = tools_dir

    def csv_lookup(self, query: str) -> ToolResult:
        csv_path = os.path.join(self.tools_dir, "prices.csv")
        matches: List[Dict[str, Any]] = []
        if pd is not None:
            try:
                df = pd.read_csv(csv_path)
                mask = df["product"].str.contains(query, case=False, na=False)
                matches = df[mask].to_dict(orient="records")
            except Exception:
                matches = []
        else:
            try:
                with open(csv_path, newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        if query.lower() in str(row.get("product", "")).lower():
                            matches.append(row)
            except FileNotFoundError:
                matches = []
        return ToolResult(tool="csv_lookup", input={"query": query}, output=matches)

    def run(self, action: Optional[Dict[str, Any]]) -> Optional[ToolResult]:
        if not action:
            return None
        if action.get("type") == "csv_lookup":
            return self.csv_lookup(action.get("query", ""))
        raise ValueError(f"Unknown action type: {action}")

