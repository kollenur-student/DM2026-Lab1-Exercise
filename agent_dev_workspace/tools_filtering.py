from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def variance_filter_tool(
        threshold: float = 0.0,
    ) -> Dict[str, Any]:
        """Filter Document-Term Matrix features by variance threshold.

        Calculates per-term variance across all documents in session.feature_matrix,
        identifying terms whose variance meets or exceeds the specified threshold.

        Args:
            threshold: Minimum variance threshold for keeping a feature (default 0.0).

        Returns:
            Dict containing result_id, threshold, total_terms, n_kept, n_removed, kept_terms, removed_terms, and variances.
        """
        result_id = session.next_result_id("variance_filter")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("variance_filter_tool", {"threshold": threshold}, summary)
            return summary

        dtm_dense = session.feature_matrix.toarray()
        variances = dtm_dense.var(axis=0, ddof=0)
        terms = session.feature_names

        var_dict = {str(term): float(var) for term, var in zip(terms, variances)}

        kept_terms = [term for term, var in var_dict.items() if var >= threshold]
        removed_terms = [term for term, var in var_dict.items() if var < threshold]

        report_data = sorted(var_dict.items(), key=lambda x: x[1], reverse=True)
        report_df = pd.DataFrame(report_data, columns=["term", "variance"])

        summary = {
            "result_id": result_id,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "variances": var_dict,
        }

        session.store_result(
            "variance_filter_tool",
            {"threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    @tool
    def pearson_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        label_column: str = "category_name",
    ) -> Dict[str, Any]:
        """Compute Pearson correlation between term counts and a target class indicator.

        Args:
            target_class: Category label string to compute one-vs-rest correlation against.
            threshold: Minimum correlation threshold for feature filtering (default 0.0).
            label_column: DataFrame column containing category labels (default 'category_name').

        Returns:
            Dict containing result_id, target_class, threshold, correlations, kept_terms, and removed_terms.
        """
        result_id = session.next_result_id("pearson_filter")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        if session.dataframe is None or label_column not in session.dataframe.columns:
            summary = {
                "result_id": result_id,
                "error": f"Label column '{label_column}' not found in session.dataframe.",
            }
            session.store_result("pearson_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        labels = session.dataframe[label_column].astype(str).values
        y_binary = (labels == str(target_class)).astype(float)

        dtm_dense = session.feature_matrix.toarray()
        terms = session.feature_names

        corr_dict = {}
        for idx, term in enumerate(terms):
            x = dtm_dense[:, idx].astype(float)
            if np.std(x) == 0 or np.std(y_binary) == 0:
                r = 0.0
            else:
                r, _ = pearsonr(x, y_binary)
                r = float(r) if not np.isnan(r) else 0.0

            corr_dict[str(term)] = round(r, 4)

        kept_terms = [term for term, r in corr_dict.items() if abs(r) >= threshold]
        removed_terms = [term for term, r in corr_dict.items() if abs(r) < threshold]

        report_data = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
        report_df = pd.DataFrame(report_data, columns=["term", "pearson_r"])

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": corr_dict,
        }

        session.store_result(
            "pearson_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    @tool
    def spearman_filter_tool(
        target_class: str,
        threshold: float = 0.0,
        label_column: str = "category_name",
    ) -> Dict[str, Any]:
        """Compute Spearman rank correlation between term counts and a target class indicator.

        Args:
            target_class: Category label string to compute one-vs-rest correlation against.
            threshold: Minimum correlation threshold for feature filtering (default 0.0).
            label_column: DataFrame column containing category labels (default 'category_name').

        Returns:
            Dict containing result_id, target_class, threshold, correlations, kept_terms, and removed_terms.
        """
        result_id = session.next_result_id("spearman_filter")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        if session.dataframe is None or label_column not in session.dataframe.columns:
            summary = {
                "result_id": result_id,
                "error": f"Label column '{label_column}' not found in session.dataframe.",
            }
            session.store_result("spearman_filter_tool", {"target_class": target_class, "threshold": threshold}, summary)
            return summary

        labels = session.dataframe[label_column].astype(str).values
        y_binary = (labels == str(target_class)).astype(float)

        dtm_dense = session.feature_matrix.toarray()
        terms = session.feature_names

        corr_dict = {}
        for idx, term in enumerate(terms):
            x = dtm_dense[:, idx].astype(float)
            if np.std(x) == 0 or np.std(y_binary) == 0:
                rho = 0.0
            else:
                rho, _ = spearmanr(x, y_binary)
                rho = float(rho) if not np.isnan(rho) else 0.0

            corr_dict[str(term)] = round(rho, 4)

        kept_terms = [term for term, rho in corr_dict.items() if abs(rho) >= threshold]
        removed_terms = [term for term, rho in corr_dict.items() if abs(rho) < threshold]

        report_data = sorted(corr_dict.items(), key=lambda x: x[1], reverse=True)
        report_df = pd.DataFrame(report_data, columns=["term", "spearman_r"])

        summary = {
            "result_id": result_id,
            "target_class": target_class,
            "threshold": threshold,
            "total_terms": len(terms),
            "n_kept": len(kept_terms),
            "n_removed": len(removed_terms),
            "kept_terms": kept_terms,
            "removed_terms": removed_terms,
            "correlations": corr_dict,
        }

        session.store_result(
            "spearman_filter_tool",
            {"target_class": target_class, "threshold": threshold},
            summary,
            full_report=report_df,
        )

        return summary

    return [variance_filter_tool, pearson_filter_tool, spearman_filter_tool]
