from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def cosine_similarity_tool(
        doc1_idx: Optional[int] = None,
        doc2_idx: Optional[int] = None,
        doc1_text: Optional[str] = None,
        doc2_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Compute Cosine Similarity between two documents (by index or raw text).

        Args:
            doc1_idx: Zero-based row index of the first document in session.feature_matrix.
            doc2_idx: Zero-based row index of the second document in session.feature_matrix.
            doc1_text: Optional raw text of first document (vectorized using session vectorizer).
            doc2_text: Optional raw text of second document (vectorized using session vectorizer).

        Returns:
            Dict containing result_id, doc1_idx, doc2_idx, and similarity score.
        """
        result_id = session.next_result_id("cosine_similarity")

        v1, v2 = None, None

        if doc1_idx is not None and session.feature_matrix is not None:
            if 0 <= doc1_idx < session.feature_matrix.shape[0]:
                v1 = session.feature_matrix[doc1_idx].toarray()

        if doc2_idx is not None and session.feature_matrix is not None:
            if 0 <= doc2_idx < session.feature_matrix.shape[0]:
                v2 = session.feature_matrix[doc2_idx].toarray()

        vectorizer = session.artifacts.get("count_vectorizer")

        if v1 is None and doc1_text is not None and vectorizer is not None:
            v1 = vectorizer.transform([doc1_text]).toarray()

        if v2 is None and doc2_text is not None and vectorizer is not None:
            v2 = vectorizer.transform([doc2_text]).toarray()

        if v1 is None or v2 is None:
            summary = {
                "result_id": result_id,
                "error": "Could not obtain document vectors. Provide valid indices (after build_dtm) or raw text.",
            }
            session.store_result("cosine_similarity_tool", {}, summary)
            return summary

        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 == 0.0 or norm2 == 0.0:
            sim_score = 0.0
        else:
            sim_matrix = cosine_similarity(v1, v2)
            sim_score = float(sim_matrix[0, 0])

        sim_score = round(sim_score, 4)

        summary = {
            "result_id": result_id,
            "doc1_idx": doc1_idx,
            "doc2_idx": doc2_idx,
            "similarity": sim_score,
        }

        session.store_result(
            "cosine_similarity_tool",
            {"doc1_idx": doc1_idx, "doc2_idx": doc2_idx, "doc1_text": doc1_text, "doc2_text": doc2_text},
            summary,
        )

        return summary

    @tool
    def feature_correlation_matrix_tool(
        top_k: int = 20,
    ) -> Dict[str, Any]:
        """Compute and plot feature correlation matrix for top variance terms in session.feature_matrix.

        Args:
            top_k: Number of top-variance terms to include in correlation matrix (default 20).

        Returns:
            Dict containing result_id, top_terms, and correlation_matrix.
        """
        result_id = session.next_result_id("feature_correlation_matrix")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("feature_correlation_matrix_tool", {"top_k": top_k}, summary)
            return summary

        dtm_dense = session.feature_matrix.toarray()
        terms = np.array(session.feature_names)

        # Compute variance per term across documents
        variances = dtm_dense.var(axis=0, ddof=0)

        # Sort terms descending by variance
        sorted_indices = np.argsort(variances)[::-1]
        top_indices = sorted_indices[:top_k]

        top_terms = terms[top_indices].tolist()
        top_matrix = dtm_dense[:, top_indices]

        # Calculate Pearson correlation matrix across top variance columns
        if top_matrix.shape[1] > 1:
            corr_matrix = np.corrcoef(top_matrix, rowvar=False)
            # Handle potential NaNs in constant vectors
            corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        else:
            corr_matrix = np.array([[1.0]])

        corr_matrix_rounded = np.round(corr_matrix, 4).tolist()

        # Plot annotated heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(top_terms) * 0.8), max(5, len(top_terms) * 0.8)))
        sns.heatmap(
            corr_matrix,
            xticklabels=top_terms,
            yticklabels=top_terms,
            annot=True,
            fmt=".2f",
            cmap="coolwarm",
            vmin=-1.0,
            vmax=1.0,
            ax=ax,
        )
        ax.set_title(f"Feature Correlation Heatmap (Top {len(top_terms)} Terms by Variance)")
        plt.tight_layout()

        # Set matplotlib figure on session.pending_figure
        session.pending_figure = fig

        summary = {
            "result_id": result_id,
            "top_terms": top_terms,
            "correlation_matrix": corr_matrix_rounded,
        }

        session.store_result("feature_correlation_matrix_tool", {"top_k": top_k}, summary)

        return summary

    return [cosine_similarity_tool, feature_correlation_matrix_tool]
