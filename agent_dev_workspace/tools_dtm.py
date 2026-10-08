from typing import Dict, List, Any, Optional
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def build_dtm_tool(
        text_column: str = "text",
        label_column: str = "category_name",
        ngram_range: List[int] = [1, 1],
        min_df: int = 1,
        max_df: float = 1.0,
        lowercase: bool = True,
    ) -> Dict[str, Any]:
        """Build the Document-Term Matrix (DTM) from text documents in session state.

        Args:
            text_column: Column name containing document text (default 'text').
            label_column: Column name containing category labels (default 'category_name').
            ngram_range: Lower and upper boundary of range of n-values for n-grams (default [1, 1]).
            min_df: Minimum document frequency threshold for vocabulary inclusion (default 1).
            max_df: Maximum document frequency threshold for vocabulary inclusion (default 1.0).
            lowercase: Whether to convert all characters to lowercase before tokenizing (default True).

        Returns:
            Dict containing result_id, n_documents, vocabulary_size, non_zero, total_elements, and sparsity_pct.
        """
        result_id = session.next_result_id("build_dtm")

        if session.dataframe is None:
            summary = {"result_id": result_id, "error": "No DataFrame is loaded in session state."}
            session.store_result("build_dtm_tool", {}, summary)
            return summary

        df = session.dataframe

        if text_column not in df.columns:
            summary = {"result_id": result_id, "error": f"Column '{text_column}' not found in DataFrame."}
            session.store_result("build_dtm_tool", {"text_column": text_column}, summary)
            return summary

        ngram_tuple = tuple(ngram_range)
        texts = df[text_column].fillna("").astype(str).tolist()

        vectorizer = CountVectorizer(
            ngram_range=ngram_tuple,
            min_df=min_df,
            max_df=max_df,
            lowercase=lowercase,
        )

        dtm = vectorizer.fit_transform(texts)
        feature_names = list(vectorizer.get_feature_names_out())

        session.feature_matrix = dtm
        session.feature_names = feature_names
        session.artifacts["count_vectorizer"] = vectorizer

        if label_column in df.columns:
            session.set_labels(df[label_column].values)

        n_documents, vocabulary_size = dtm.shape
        non_zero = int(dtm.nnz)
        total_elements = int(n_documents * vocabulary_size)

        sparsity_pct = 0.0
        if total_elements > 0:
            sparsity_pct = round(100.0 * (1.0 - (non_zero / total_elements)), 4)

        summary = {
            "result_id": result_id,
            "n_documents": n_documents,
            "vocabulary_size": vocabulary_size,
            "non_zero": non_zero,
            "total_elements": total_elements,
            "sparsity_pct": sparsity_pct,
            "vocabulary": feature_names,
        }

        session.store_result(
            "build_dtm_tool",
            {
                "text_column": text_column,
                "label_column": label_column,
                "ngram_range": ngram_range,
                "min_df": min_df,
                "max_df": max_df,
                "lowercase": lowercase,
            },
            summary,
        )

        return summary

    @tool
    def term_frequency_tool(
        top_k: Optional[int] = 20,
    ) -> Dict[str, Any]:
        """Aggregate term frequencies across all documents in the document-term matrix.

        Args:
            top_k: Optional number of top terms to highlight in summary (default 20).

        Returns:
            Dict containing result_id, total_terms, and term_frequencies dictionary.
        """
        result_id = session.next_result_id("term_frequency")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("term_frequency_tool", {"top_k": top_k}, summary)
            return summary

        counts = np.asarray(session.feature_matrix.sum(axis=0)).flatten()
        terms = session.feature_names

        term_freqs = {str(term): int(count) for term, count in zip(terms, counts)}

        report_df = pd.DataFrame(
            [{"term": t, "frequency": f} for t, f in sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)]
        )

        top_terms_summary = dict(sorted(term_freqs.items(), key=lambda x: x[1], reverse=True)[:top_k]) if top_k else term_freqs

        summary = {
            "result_id": result_id,
            "total_terms": len(terms),
            "term_frequencies": term_freqs,
            "top_terms": top_terms_summary,
        }

        session.store_result("term_frequency_tool", {"top_k": top_k}, summary, full_report=report_df)

        return summary

    @tool
    def dtm_heatmap_tool(
        n_terms: int = 20,
        n_documents: int = 20,
    ) -> Dict[str, Any]:
        """Plot a heatmap slice of the raw Document-Term Matrix in session state.

        Args:
            n_terms: Number of terms/columns to include from DTM (default 20).
            n_documents: Number of documents/rows to include from DTM (default 20).

        Returns:
            Dict containing result_id, n_documents_sliced, n_terms_sliced, terms, and sliced matrix.
        """
        result_id = session.next_result_id("dtm_heatmap")

        if session.feature_matrix is None or session.feature_names is None:
            summary = {
                "result_id": result_id,
                "error": "No feature_matrix or feature_names in session state. Run build_dtm_tool first.",
            }
            session.store_result("dtm_heatmap_tool", {"n_terms": n_terms, "n_documents": n_documents}, summary)
            return summary

        # Positional slicing [0:n_documents, 0:n_terms]
        matrix_slice = session.feature_matrix[0:n_documents, 0:n_terms].toarray()
        sliced_terms = session.feature_names[0:n_terms]
        sliced_docs = [f"Doc {i}" for i in range(matrix_slice.shape[0])]

        # Plot Seaborn heatmap
        fig, ax = plt.subplots(figsize=(max(6, len(sliced_terms) * 0.8), max(4, len(sliced_docs) * 0.4)))
        sns.heatmap(
            matrix_slice,
            xticklabels=sliced_terms,
            yticklabels=sliced_docs,
            annot=True,
            fmt="d",
            cmap="YlGnBu",
            ax=ax,
        )
        ax.set_title("Document-Term Matrix Heatmap")
        ax.set_xlabel("Terms")
        ax.set_ylabel("Documents")
        plt.tight_layout()

        # Set matplotlib figure on session.pending_figure
        session.pending_figure = fig

        summary = {
            "result_id": result_id,
            "n_documents_sliced": matrix_slice.shape[0],
            "n_terms_sliced": matrix_slice.shape[1],
            "terms": sliced_terms,
            "matrix": matrix_slice.tolist(),
        }

        session.store_result(
            "dtm_heatmap_tool",
            {"n_terms": n_terms, "n_documents": n_documents},
            summary,
        )

        return summary

    return [build_dtm_tool, term_frequency_tool, dtm_heatmap_tool]
