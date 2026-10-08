import os
from pathlib import Path
from typing import Dict, List, Any, Optional
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def list_files_tool(subdirectory: str = "") -> Dict[str, Any]:
        """List files and subdirectories in the specified workspace directory.

        Args:
            subdirectory: Relative path from workspace root to list contents for (e.g., 'newdataset').

        Returns:
            Dict containing 'result_id', 'directories' (list of folder names), and 'files' (list of file names).
        """
        result_id = session.next_result_id("list_files")
        
        target_dir = Path(subdirectory) if subdirectory else Path(".")
        
        if not target_dir.exists() or not target_dir.is_dir():
            summary = {
                "result_id": result_id,
                "directories": [],
                "files": [],
                "error": f"Directory '{subdirectory}' does not exist or is not a directory."
            }
            session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
            return summary

        directories = []
        files = []

        for item in target_dir.iterdir():
            if item.name.startswith("."):
                continue
            if item.is_dir():
                directories.append(item.name)
            elif item.is_file():
                files.append(item.name)

        directories.sort()
        files.sort()

        summary = {
            "result_id": result_id,
            "directories": directories,
            "files": files
        }

        session.store_result("list_files_tool", {"subdirectory": subdirectory}, summary)
        return summary

    @tool
    def inspect_data_tool(n: int = 5, position: str = "head") -> Dict[str, Any]:
        """Inspect a preview of rows from the working DataFrame.

        Args:
            n: Number of rows to inspect (default 5).
            position: Position to inspect from, either 'head' or 'tail' (default 'head').

        Returns:
            Dict containing 'result_id', 'n_rows_total', 'n_rows_inspected', 'columns', and 'rows'.
        """
        result_id = session.next_result_id("inspect_data")

        if session.dataframe is None:
            summary = {
                "result_id": result_id,
                "error": "No DataFrame is loaded in session state."
            }
            session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
            return summary

        df = session.dataframe
        if position.lower() == "tail":
            subset = df.tail(n)
        else:
            subset = df.head(n)

        rows = subset.fillna("").to_dict(orient="records")

        summary = {
            "result_id": result_id,
            "n_rows_total": len(df),
            "n_rows_inspected": len(subset),
            "columns": list(df.columns),
            "rows": rows
        }

        session.store_result("inspect_data_tool", {"n": n, "position": position}, summary)
        return summary

    @tool
    def check_missing_tool(text_column: str = "text") -> Dict[str, Any]:
        """Check for missing or empty text values in the specified column.

        Args:
            text_column: Name of the column to check for missing or empty values (default 'text').

        Returns:
            Dict containing 'result_id', 'n_rows_total', 'n_missing', 'missing_pct', and 'missing_indices'.
        """
        result_id = session.next_result_id("check_missing")

        if session.dataframe is None:
            summary = {
                "result_id": result_id,
                "error": "No DataFrame is loaded in session state."
            }
            session.store_result("check_missing_tool", {"text_column": text_column}, summary)
            return summary

        df = session.dataframe

        if text_column not in df.columns:
            summary = {
                "result_id": result_id,
                "error": f"Column '{text_column}' not found in DataFrame."
            }
            session.store_result("check_missing_tool", {"text_column": text_column}, summary)
            return summary

        col_series = df[text_column]
        is_null = col_series.isna() | col_series.isnull()
        is_empty = col_series.astype(str).str.strip() == ""
        missing_mask = is_null | is_empty

        missing_indices = df.index[missing_mask].tolist()
        n_missing = len(missing_indices)
        n_total = len(df)
        missing_pct = round(100.0 * (n_missing / n_total), 4) if n_total > 0 else 0.0

        summary = {
            "result_id": result_id,
            "n_rows_total": n_total,
            "n_missing": n_missing,
            "missing_pct": missing_pct,
            "missing_indices": missing_indices
        }

        session.store_result("check_missing_tool", {"text_column": text_column}, summary)
        return summary

    @tool
    def check_duplicates_tool(subset_column: str = "text", drop: bool = False) -> Dict[str, Any]:
        """Check for (and optionally drop) duplicate documents.

        Args:
            subset_column: Column name to check for duplicate values (default 'text').
            drop: If True, drop all copies of duplicated rows using keep=False (default False).

        Returns:
            Dict containing 'result_id', 'n_rows_before', 'n_duplicates', 'dropped', 'n_rows_after', and 'n_dropped'.
        """
        result_id = session.next_result_id("check_duplicates")

        if session.dataframe is None:
            summary = {
                "result_id": result_id,
                "error": "No DataFrame is loaded in session state."
            }
            session.store_result("check_duplicates_tool", {"subset_column": subset_column, "drop": drop}, summary)
            return summary

        df = session.dataframe

        if subset_column not in df.columns:
            summary = {
                "result_id": result_id,
                "error": f"Column '{subset_column}' not found in DataFrame."
            }
            session.store_result("check_duplicates_tool", {"subset_column": subset_column, "drop": drop}, summary)
            return summary

        n_rows_before = len(df)
        n_duplicates = int(df.duplicated(subset=[subset_column], keep="first").sum())

        n_rows_after = n_rows_before
        n_dropped = 0

        if drop:
            cleaned_df = df.drop_duplicates(subset=[subset_column], keep=False)
            session.dataframe = cleaned_df.reset_index(drop=True)
            n_rows_after = len(session.dataframe)
            n_dropped = n_rows_before - n_rows_after

        summary = {
            "result_id": result_id,
            "n_rows_before": n_rows_before,
            "n_duplicates": n_duplicates,
            "dropped": drop,
            "n_rows_after": n_rows_after,
            "n_dropped": n_dropped
        }

        session.store_result("check_duplicates_tool", {"subset_column": subset_column, "drop": drop}, summary)
        return summary

    @tool
    def sample_data_tool(
        n: Optional[int] = None,
        frac: Optional[float] = None,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Subsample the working DataFrame in session state.

        Takes a random sample of rows from session.dataframe using either a fixed number of rows
        or a fraction of the total rows, and updates session.dataframe with the sampled subset.

        Args:
            n: Optional number of items/rows to return.
            frac: Optional fraction of axis items to return (e.g., 0.5 for 50%).
            random_state: Optional seed for the random number generator for reproducibility.

        Returns:
            A dictionary containing result_id, previous_row_count, sampled_row_count, and parameters used.
        """
        if session.dataframe is None:
            return {"status": "error", "message": "No dataset loaded in session.dataframe."}

        previous_count = len(session.dataframe)

        if n is None and frac is None:
            n = min(5, previous_count)

        sampled_df = session.dataframe.sample(n=n, frac=frac, random_state=random_state)
        session.dataframe = sampled_df.reset_index(drop=True)

        sampled_count = len(session.dataframe)
        result_id = session.next_result_id("sample_data")

        summary = {
            "result_id": result_id,
            "previous_row_count": previous_count,
            "sampled_row_count": sampled_count,
            "n": n,
            "frac": frac,
            "random_state": random_state,
        }

        session.store_result("sample_data_tool", {"n": n, "frac": frac, "random_state": random_state}, summary)
        return summary

    @tool
    def describe_data_tool(
        text_column: str = "text",
        category_column: str = "category_name",
    ) -> Dict[str, Any]:
        """Describe document text length stats overall and per category, and plot box plot.

        Args:
            text_column: Column containing document text (default 'text').
            category_column: Column containing category labels (default 'category_name').

        Returns:
            Dict containing result_id, text_length summary stats (overall and per category).
        """
        result_id = session.next_result_id("describe_data")

        if session.dataframe is None:
            summary = {"result_id": result_id, "error": "No DataFrame is loaded in session state."}
            session.store_result("describe_data_tool", {"text_column": text_column, "category_column": category_column}, summary)
            return summary

        df = session.dataframe

        if text_column not in df.columns:
            summary = {"result_id": result_id, "error": f"Column '{text_column}' not found in DataFrame."}
            session.store_result("describe_data_tool", {"text_column": text_column, "category_column": category_column}, summary)
            return summary

        # Compute text length
        df["text_length"] = df[text_column].fillna("").apply(len)

        # Overall describe stats
        overall_stats = df["text_length"].describe().to_dict()

        # Per category stats
        per_category_stats = {}
        if category_column in df.columns:
            for cat, group in df.groupby(category_column):
                per_category_stats[str(cat)] = group["text_length"].describe().to_dict()

        # Create Seaborn boxplot figure
        fig, ax = plt.subplots(figsize=(8, 5))
        if category_column in df.columns:
            sns.boxplot(data=df, x=category_column, y="text_length", ax=ax)
            ax.set_title(f"Text Length Distribution by {category_column}")
            ax.set_xlabel(category_column)
        else:
            sns.boxplot(data=df, y="text_length", ax=ax)
            ax.set_title("Text Length Distribution Overall")

        ax.set_ylabel("Text Length (character count)")
        plt.tight_layout()

        # Store pending figure on session
        session.pending_figure = fig

        summary = {
            "result_id": result_id,
            "overall_stats": overall_stats,
            "per_category_stats": per_category_stats,
        }

        session.store_result(
            "describe_data_tool",
            {"text_column": text_column, "category_column": category_column},
            summary,
        )

        return summary

    return [
        list_files_tool,
        inspect_data_tool,
        check_missing_tool,
        check_duplicates_tool,
        sample_data_tool,
        describe_data_tool,
    ]
