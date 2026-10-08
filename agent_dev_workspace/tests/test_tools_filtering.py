import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_dtm import make_tools as make_dtm_tools
from tools_filtering import make_tools as make_filtering_tools


def test_variance_filter_tool_known_answer_5():
    """Test variance_filter_tool against known answer 5 in TEST_FIXTURE.md."""
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

    filtering_tools = make_filtering_tools(session)
    variance_filter_tool = next(t for t in filtering_tools if t.name == "variance_filter_tool")

    res = variance_filter_tool.invoke({"threshold": 0.15})

    variances = res["variances"]
    assert np.isclose(variances["alpha"], 1.1875, atol=1e-4)
    assert np.isclose(variances["always"], 0.109375, atol=1e-4)
    assert np.isclose(variances["beta"], 0.234375, atol=1e-4)
    assert np.isclose(variances["delta"], 0.25, atol=1e-4)
    assert np.isclose(variances["gamma"], 1.109375, atol=1e-4)

    assert set(res["kept_terms"]) == {"alpha", "beta", "delta", "gamma"}
    assert set(res["removed_terms"]) == {"always"}
    assert res["n_kept"] == 4
    assert res["n_removed"] == 1


def test_pearson_filter_tool_known_answer_6():
    """Test pearson_filter_tool against known answer 6 in TEST_FIXTURE.md."""
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

    filtering_tools = make_filtering_tools(session)
    pearson_filter_tool = next(t for t in filtering_tools if t.name == "pearson_filter_tool")

    res = pearson_filter_tool.invoke({
        "target_class": "catB",
        "label_column": "category_name"
    })

    corrs = res["correlations"]
    assert np.isclose(corrs["alpha"], -0.6882, atol=1e-3)
    assert np.isclose(corrs["always"], 0.3780, atol=1e-3)
    assert np.isclose(corrs["beta"], -0.7746, atol=1e-3)
    assert np.isclose(corrs["delta"], 1.0000, atol=1e-3)
    assert np.isclose(corrs["gamma"], 0.8307, atol=1e-3)


def test_spearman_filter_tool_known_answer_6():
    """Test spearman_filter_tool against known answer 6 in TEST_FIXTURE.md."""
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

    filtering_tools = make_filtering_tools(session)
    spearman_filter_tool = next(t for t in filtering_tools if t.name == "spearman_filter_tool")

    # Known answer #6 Spearman r for target_class="catB":
    # alpha: -0.7500
    # always: 0.3780
    # beta: -0.7746
    # delta: 1.0000
    # gamma: 0.9363
    res = spearman_filter_tool.invoke({
        "target_class": "catB",
        "label_column": "category_name"
    })

    corrs = res["correlations"]
    assert np.isclose(corrs["alpha"], -0.7500, atol=1e-3)
    assert np.isclose(corrs["always"], 0.3780, atol=1e-3)
    assert np.isclose(corrs["beta"], -0.7746, atol=1e-3)
    assert np.isclose(corrs["delta"], 1.0000, atol=1e-3)
    assert np.isclose(corrs["gamma"], 0.9363, atol=1e-3)
