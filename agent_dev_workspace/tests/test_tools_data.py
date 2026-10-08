import pytest
import numpy as np
from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools
from tools_data import make_tools


def test_list_files_tool():
    """Test list_files_tool against repo contents (known answer 11 in TEST_FIXTURE.md)."""
    session = SessionState()
    tools = make_tools(session)
    list_files_tool = next(t for t in tools if t.name == "list_files_tool")

    result = list_files_tool.invoke({"subdirectory": "newdataset"})

    assert "result_id" in result
    assert result["directories"] == []
    assert result["files"] == ["Reddit-stock-sentiment.csv"]


def test_inspect_data_tool():
    """Test inspect_data_tool with sample_fixture.csv dataset loaded."""
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    inspect_data_tool = next(t for t in tools if t.name == "inspect_data_tool")

    result_head = inspect_data_tool.invoke({"n": 3, "position": "head"})
    assert result_head["n_rows_total"] == 8
    assert result_head["n_rows_inspected"] == 3
    assert len(result_head["rows"]) == 3
    assert result_head["rows"][0]["text"] == "always alpha alpha alpha beta"

    result_tail = inspect_data_tool.invoke({"n": 2, "position": "tail"})
    assert result_tail["n_rows_inspected"] == 2
    assert len(result_tail["rows"]) == 2
    assert result_tail["rows"][-1]["text"] == ""


def test_check_missing_tool():
    """Test check_missing_tool against known answer 2 in TEST_FIXTURE.md."""
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_missing_tool = next(t for t in tools if t.name == "check_missing_tool")

    result = check_missing_tool.invoke({"text_column": "text"})

    assert result["n_rows_total"] == 8
    assert result["n_missing"] == 1
    assert result["missing_indices"] == [7]


def test_check_duplicates_tool():
    """Test check_duplicates_tool against known answer 3 in TEST_FIXTURE.md."""
    session = SessionState()
    
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")
    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    tools = make_tools(session)
    check_duplicates_tool = next(t for t in tools if t.name == "check_duplicates_tool")

    check_res = check_duplicates_tool.invoke({"subset_column": "text", "drop": False})
    assert check_res["n_rows_before"] == 8
    assert check_res["n_duplicates"] == 1
    assert check_res["dropped"] is False
    assert len(session.dataframe) == 8

    drop_res = check_duplicates_tool.invoke({"subset_column": "text", "drop": True})
    assert drop_res["n_rows_before"] == 8
    assert drop_res["dropped"] is True
    assert drop_res["n_rows_after"] == 6
    assert drop_res["n_dropped"] == 2
    assert len(session.dataframe) == 6


def test_sample_data_tool_known_answer_13():
    """Test sample_data_tool against known answer 13 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    sample_data_tool = next(t for t in data_tools if t.name == "sample_data_tool")

    result = sample_data_tool.invoke({"n": 4, "random_state": 42})

    assert result["sampled_row_count"] == 4
    assert result["previous_row_count"] == 8

    expected_texts = [
        "always alpha alpha beta",
        "always gamma delta",
        "always alpha alpha alpha beta",
        "",
    ]

    actual_texts = session.dataframe["text"].tolist()
    assert actual_texts == expected_texts


def test_describe_data_tool_known_answer_18():
    """Test describe_data_tool against known answer 18 in TEST_FIXTURE.md."""
    session = SessionState()

    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    data_tools = make_tools(session)
    describe_data_tool = next(t for t in data_tools if t.name == "describe_data_tool")

    result = describe_data_tool.invoke({
        "text_column": "text",
        "category_column": "category_name"
    })

    # Verify pending figure was created
    assert session.pending_figure is not None

    # Verify per-document text_length character counts
    expected_lengths = [29, 23, 17, 30, 24, 18, 18, 0]
    actual_lengths = session.dataframe["text_length"].tolist()
    assert actual_lengths == expected_lengths

    # Verify overall stats (Known answer #18: count=8, mean=19.875, std=9.4330, min=0, 25%=17.75, 50%=20.5, 75%=25.25, max=30)
    overall = result["overall_stats"]
    assert overall["count"] == 8
    assert np.isclose(overall["mean"], 19.875, atol=1e-3)
    assert np.isclose(overall["std"], 9.4330, atol=1e-3)
    assert overall["min"] == 0
    assert np.isclose(overall["25%"], 17.75, atol=1e-3)
    assert np.isclose(overall["50%"], 20.5, atol=1e-3)
    assert np.isclose(overall["75%"], 25.25, atol=1e-3)
    assert overall["max"] == 30

    # Verify catA stats (count=4, mean=17.25, std=12.5, min=0, 25%=12.75, 50%=20.0, 75%=24.5, max=29)
    catA = result["per_category_stats"]["catA"]
    assert catA["count"] == 4
    assert np.isclose(catA["mean"], 17.25, atol=1e-3)
    assert np.isclose(catA["std"], 12.5, atol=1e-3)
    assert catA["min"] == 0
    assert np.isclose(catA["25%"], 12.75, atol=1e-3)
    assert np.isclose(catA["50%"], 20.0, atol=1e-3)
    assert np.isclose(catA["75%"], 24.5, atol=1e-3)
    assert catA["max"] == 29

    # Verify catB stats (count=4, mean=22.5, std=5.7446, min=18, 25%=18.0, 50%=21.0, 75%=25.5, max=30)
    catB = result["per_category_stats"]["catB"]
    assert catB["count"] == 4
    assert np.isclose(catB["mean"], 22.5, atol=1e-3)
    assert np.isclose(catB["std"], 5.7446, atol=1e-3)
    assert catB["min"] == 18
    assert np.isclose(catB["25%"], 18.0, atol=1e-3)
    assert np.isclose(catB["50%"], 21.0, atol=1e-3)
    assert np.isclose(catB["75%"], 25.5, atol=1e-3)
    assert catB["max"] == 30
