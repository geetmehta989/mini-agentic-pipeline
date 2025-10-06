import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

import pandas as pd

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
        df = pd.read_csv(csv_path)
        # simple contains match on product name (case-insensitive)
        mask = df["product"].str.contains(query, case=False, na=False)
        matches = df[mask].to_dict(orient="records")
        return ToolResult(tool="csv_lookup", input={"query": query}, output=matches)

    def run(self, action: Optional[Dict[str, Any]]) -> Optional[ToolResult]:
        if not action:
            return None
        if action.get("type") == "csv_lookup":
            return self.csv_lookup(action.get("query", ""))
        raise ValueError(f"Unknown action type: {action}")

