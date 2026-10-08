from agent_pipeline.session_state import SessionState
from agent_pipeline.workspace_loader import load_workspace_tools


def test_load_dataset_tool():
    # 1. Initialize a single SessionState instance
    session = SessionState()

    # 2. Load premade tools from the "premade_tools" directory
    premade_tools, _ = load_workspace_tools("premade_tools", session)
    load_dataset_tool = next(t for t in premade_tools if t.name == "load_dataset_tool")

    # 3. Invoke load_dataset_tool on agent_dev/sample_fixture.csv
    result = load_dataset_tool.invoke({
        "file_path": "agent_dev/sample_fixture.csv",
        "text_column": "text",
        "label_column": "label"
    })

    # 4. Assert against known answer #1 in TEST_FIXTURE.md:
    # Exactly 8 documents total, and counts_per_category equal to {"catA": 4, "catB": 4}
    assert result["n_documents"] == 8
    assert result["counts_per_category"] == {"catA": 4, "catB": 4}
