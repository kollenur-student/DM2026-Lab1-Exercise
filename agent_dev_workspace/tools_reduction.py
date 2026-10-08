from typing import Dict, List, Any, Optional, Literal
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import LabelBinarizer
import umap
from langchain_core.tools import tool


def make_tools(session):
    @tool
    def reduce_dimensions_tool(
        method: Literal["pca", "tsne", "umap"] = "pca",
        perplexity: float = 30.0,
        n_neighbors: int = 15,
        random_state: Optional[int] = 42,
    ) -> Dict[str, Any]:
        """Reduce Document-Term Matrix dimensions to 2D using PCA, t-SNE, or UMAP, and plot scatter.

        Args:
            method: Dimensionality reduction algorithm ('pca', 'tsne', or 'umap').
            perplexity: Perplexity parameter for t-SNE (default 30.0).
            n_neighbors: Number of neighbors parameter for UMAP (default 15).
            random_state: Optional seed for reproducible output (default 42).

        Returns:
            Dict containing result_id, method, coordinates, and explained_variance_ratio (PCA).
        """
        result_id = session.next_result_id("reduce_dimensions")

        if session.feature_matrix is None:
            summary = {"result_id": result_id, "error": "No feature_matrix in session state. Run build_dtm_tool first."}
            session.store_result("reduce_dimensions_tool", {"method": method}, summary)
            return summary

        X = session.feature_matrix.toarray()
        n_samples = X.shape[0]

        explained_variance_ratio = None

        if method == "pca":
            reducer = PCA(n_components=2, random_state=random_state)
            coords = reducer.fit_transform(X)
            explained_variance_ratio = [round(float(v), 4) for v in reducer.explained_variance_ratio_]
        elif method == "tsne":
            eff_perp = min(perplexity, max(1.0, float(n_samples - 1)))
            reducer = TSNE(n_components=2, perplexity=eff_perp, random_state=random_state)
            coords = reducer.fit_transform(X)
        elif method == "umap":
            eff_neighbors = min(n_neighbors, max(2, n_samples - 1))
            reducer = umap.UMAP(n_components=2, n_neighbors=eff_neighbors, random_state=random_state)
            coords = reducer.fit_transform(X)
        else:
            summary = {"result_id": result_id, "error": f"Unsupported method '{method}'."}
            session.store_result("reduce_dimensions_tool", {"method": method}, summary)
            return summary

        coords_list = [[round(float(pt[0]), 4), round(float(pt[1]), 4)] for pt in coords]

        df = session.dataframe
        if df is not None and "category_name" in df.columns:
            categories = df["category_name"].astype(str).tolist()
        elif df is not None and "label" in df.columns:
            categories = df["label"].astype(str).tolist()
        elif session.labels is not None:
            categories = [str(lbl) for lbl in session.labels]
        else:
            categories = ["Doc"] * n_samples

        fig, ax = plt.subplots(figsize=(8, 6))
        plot_df = pd.DataFrame({
            "Dim 1": [pt[0] for pt in coords],
            "Dim 2": [pt[1] for pt in coords],
            "Category": categories,
        })

        sns.scatterplot(data=plot_df, x="Dim 1", y="Dim 2", hue="Category", style="Category", s=100, ax=ax)
        ax.set_title(f"2D Projection via {method.upper()}")
        ax.set_xlabel("Dimension 1")
        ax.set_ylabel("Dimension 2")
        plt.tight_layout()

        session.pending_figure = fig
        session.artifacts[f"reduced_{method}"] = coords_list

        summary = {
            "result_id": result_id,
            "method": method,
            "n_components": 2,
            "coordinates": coords_list,
        }
        if explained_variance_ratio is not None:
            summary["explained_variance_ratio"] = explained_variance_ratio

        session.store_result(
            "reduce_dimensions_tool",
            {"method": method, "perplexity": perplexity, "n_neighbors": n_neighbors, "random_state": random_state},
            summary,
        )

        return summary

    @tool
    def binarize_labels_tool(
        label_column: str = "category_name",
    ) -> Dict[str, Any]:
        """One-hot encode/binarize categorical target labels in session state.

        Args:
            label_column: DataFrame column containing categorical labels (default 'category_name').

        Returns:
            Dict containing result_id, classes, shape, and binarized_matrix list.
        """
        result_id = session.next_result_id("binarize_labels")

        if session.dataframe is None:
            summary = {"result_id": result_id, "error": "No DataFrame loaded in session state."}
            session.store_result("binarize_labels_tool", {"label_column": label_column}, summary)
            return summary

        df = session.dataframe
        if label_column not in df.columns:
            label_column = "label" if "label" in df.columns else label_column

        if label_column not in df.columns:
            summary = {"result_id": result_id, "error": f"Column '{label_column}' not found in DataFrame."}
            session.store_result("binarize_labels_tool", {"label_column": label_column}, summary)
            return summary

        labels = df[label_column].astype(str).tolist()

        lb = LabelBinarizer()
        binarized = lb.fit_transform(labels)
        classes = [str(c) for c in lb.classes_]

        # For binary classification with 1 output column, convert to 2-column one-hot matrix
        if binarized.shape[1] == 1 and len(classes) == 2:
            binarized = np.hstack((1 - binarized, binarized))

        binarized_list = binarized.tolist()

        # Store artifacts
        session.artifacts["binarized_labels"] = binarized_list
        session.artifacts["binarizer_classes"] = classes

        summary = {
            "result_id": result_id,
            "label_column": label_column,
            "classes": classes,
            "shape": list(binarized.shape),
            "binarized_matrix": binarized_list,
        }

        session.store_result("binarize_labels_tool", {"label_column": label_column}, summary)

        return summary

    return [reduce_dimensions_tool, binarize_labels_tool]
