import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools


def test_build_dtm_tool_known_answer_4():
    """Test build_dtm_tool against known answer 4 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")

    result = build_dtm_tool.invoke({
        "text_column": "text",
        "label_column": "category_name"
    })

    expected_vocabulary = ["alpha", "always", "beta", "delta", "gamma"]
    assert session.feature_names == expected_vocabulary
    assert result["vocabulary"] == expected_vocabulary
    assert result["vocabulary_size"] == 5

    assert result["n_documents"] == 8
    assert result["non_zero"] == 21
    assert result["total_elements"] == 40
    assert np.isclose(result["sparsity_pct"], 47.5, atol=1e-3)

    assert session.feature_matrix is not None
    assert session.feature_matrix.shape == (8, 5)
    assert "count_vectorizer" in session.artifacts


def test_term_frequency_tool_known_answer_14():
    """Test term_frequency_tool against known answer 14 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    term_frequency_tool = next(t for t in dtm_tools if t.name == "term_frequency_tool")

    build_dtm_tool.invoke({"text_column": "text", "label_column": "category_name"})

    result = term_frequency_tool.invoke({"top_k": 20})

    expected_frequencies = {
        "alpha": 6,
        "always": 7,
        "beta": 3,
        "delta": 4,
        "gamma": 7,
    }

    assert result["total_terms"] == 5
    assert result["term_frequencies"] == expected_frequencies


def test_dtm_heatmap_tool_known_answer_21():
    """Test dtm_heatmap_tool against known answer 21 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    dtm_tools = make_tools(session)
    build_dtm_tool = next(t for t in dtm_tools if t.name == "build_dtm_tool")
    dtm_heatmap_tool = next(t for t in dtm_tools if t.name == "dtm_heatmap_tool")

    build_dtm_tool.invoke({"text_column": "text", "label_column": "category_name"})

    # Test full matrix default slice (20x20 defaults exceed 8x5 -> returns full 8x5 matrix)
    result_full = dtm_heatmap_tool.invoke({"n_terms": 20, "n_documents": 20})

    assert session.pending_figure is not None
    assert result_full["n_documents_sliced"] == 8
    assert result_full["n_terms_sliced"] == 5
    assert result_full["terms"] == ["alpha", "always", "beta", "delta", "gamma"]

    # Known answer #21 full matrix counts in vocabulary order ['alpha', 'always', 'beta', 'delta', 'gamma']:
    expected_full_matrix = [
        [3, 1, 1, 0, 0],  # Doc 0
        [2, 1, 1, 0, 0],  # Doc 1
        [1, 1, 1, 0, 0],  # Doc 2
        [0, 1, 0, 1, 3],  # Doc 3
        [0, 1, 0, 1, 2],  # Doc 4
        [0, 1, 0, 1, 1],  # Doc 5
        [0, 1, 0, 1, 1],  # Doc 6
        [0, 0, 0, 0, 0],  # Doc 7
    ]
    assert result_full["matrix"] == expected_full_matrix

    # Test top-left corner slice n_terms=2, n_documents=3
    result_slice = dtm_heatmap_tool.invoke({"n_terms": 2, "n_documents": 3})
    assert result_slice["n_documents_sliced"] == 3
    assert result_slice["n_terms_sliced"] == 2
    assert result_slice["terms"] == ["alpha", "always"]
    assert result_slice["matrix"] == [[3, 1], [2, 1], [1, 1]]
