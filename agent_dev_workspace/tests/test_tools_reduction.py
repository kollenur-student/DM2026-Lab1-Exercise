import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_reduction import make_tools as make_reduction_tools


def test_reduce_dimensions_tool_known_answers_7_8_9():
    """Test reduce_dimensions_tool against known answers 7 (PCA), 8 (t-SNE), and 9 (UMAP) in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_dtm_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    build_dtm_tool.invoke({"text_column": "text", "label_column": "category_name"})

    reduction_tools = make_reduction_tools(session)
    reduce_dimensions_tool = next(t for t in reduction_tools if t.name == "reduce_dimensions_tool")

    res_pca = reduce_dimensions_tool.invoke({"method": "pca"})
    assert session.pending_figure is not None
    assert res_pca["method"] == "pca"
    assert np.isclose(res_pca["explained_variance_ratio"][0], 0.7532, atol=1e-3)
    assert np.isclose(res_pca["explained_variance_ratio"][1], 0.1965, atol=1e-3)

    expected_pca_coords = [
        [2.3669, 0.8871],
        [1.7039, 0.2642],
        [1.0408, -0.3587],
        [-2.0759, 1.0129],
        [-1.4554, 0.3405],
        [-0.8349, -0.3319],
        [-0.8349, -0.3319],
        [0.0894, -1.4822],
    ]
    np.testing.assert_allclose(res_pca["coordinates"], expected_pca_coords, atol=1e-2)

    res_tsne = reduce_dimensions_tool.invoke({
        "method": "tsne",
        "perplexity": 2.0,
        "random_state": 42
    })
    assert session.pending_figure is not None
    assert res_tsne["method"] == "tsne"
    assert len(res_tsne["coordinates"]) == 8

    res_umap = reduce_dimensions_tool.invoke({
        "method": "umap",
        "n_neighbors": 3,
        "random_state": 42
    })
    assert session.pending_figure is not None
    assert res_umap["method"] == "umap"
    assert len(res_umap["coordinates"]) == 8


def test_binarize_labels_tool_custom():
    """Test binarize_labels_tool on sample_fixture.csv dataset."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    reduction_tools = make_reduction_tools(session)
    binarize_labels_tool = next(t for t in reduction_tools if t.name == "binarize_labels_tool")

    res = binarize_labels_tool.invoke({"label_column": "category_name"})

    assert res["classes"] == ["catA", "catB"]
    assert res["shape"] == [8, 2]

    # sample_fixture.csv rows 0..2,7 are catA [1,0]; rows 3..6 are catB [0,1]
    expected_matrix = [
        [1, 0],  # Row 0: catA
        [1, 0],  # Row 1: catA
        [1, 0],  # Row 2: catA
        [0, 1],  # Row 3: catB
        [0, 1],  # Row 4: catB
        [0, 1],  # Row 5: catB
        [0, 1],  # Row 6: catB
        [1, 0],  # Row 7: catA
    ]
    assert res["binarized_matrix"] == expected_matrix
    assert session.artifacts["binarized_labels"] == expected_matrix
    assert session.artifacts["binarizer_classes"] == ["catA", "catB"]
