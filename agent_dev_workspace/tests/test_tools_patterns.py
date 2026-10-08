import pytest
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_patterns import make_tools as make_patterns_tools


def test_mine_patterns_tool_known_answers_17_and_20():
    """Test mine_patterns_tool against known answers 17 and 20 in TEST_FIXTURE.md."""
    session = SessionState()

    # Load dataset into session
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    patterns_tools = make_patterns_tools(session)
    mine_patterns_tool = next(t for t in patterns_tools if t.name == "mine_patterns_tool")

    # 1. Test filtering_method="variance" edge case: 0 terms survive
    res_var = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "variance",
        "algorithm": "fpgrowth"
    })
    assert res_var["n_patterns"] == 0
    assert "no terms survived filtering" in res_var["note"].lower()

    # 2. Known answer #17: filtering_method="tfidf", algorithm="fpgrowth"
    # catA -> keeps alpha -> {alpha} with support 3
    res_tfidf_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_catA["n_patterns"] == 1
    assert res_tfidf_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_tfidf_catA["patterns"][0]["support"] == 3

    # catB -> keeps gamma -> {gamma} with support 4
    res_tfidf_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "fpgrowth",
        "min_sup": 1
    })
    assert res_tfidf_catB["n_patterns"] == 1
    assert res_tfidf_catB["patterns"][0]["pattern"] == ["gamma"]
    assert res_tfidf_catB["patterns"][0]["support"] == 4

    # 3. Known answer #20: algorithm="topk" at k=1
    res_topk_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "topk",
        "k": 1
    })
    assert res_topk_catA["n_patterns"] == 1
    assert res_topk_catA["patterns"][0]["pattern"] == ["alpha"]
    assert res_topk_catA["patterns"][0]["support"] == 3

    # Known answer #20: algorithm="maxfpgrowth" -> returns 0 patterns for both categories
    res_max_catA = mine_patterns_tool.invoke({
        "category_name": "catA",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catA["n_patterns"] == 0
    assert res_max_catA["patterns"] == []

    res_max_catB = mine_patterns_tool.invoke({
        "category_name": "catB",
        "filtering_method": "tfidf",
        "algorithm": "maxfpgrowth",
        "min_sup": 1
    })
    assert res_max_catB["n_patterns"] == 0
    assert res_max_catB["patterns"] == []
