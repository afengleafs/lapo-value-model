from lapo_value_model.task_families import classify_task, task_prompt


def test_representative_task_mapping() -> None:
    assert classify_task("put the apple into the bowl") == "container_transfer"
    assert classify_task("open the microwave door") == "hinged_open_close"
    assert classify_task("close the drawer") == "slidable_open_close"
    assert classify_task("wipe the table clean") == "clean"
    assert classify_task("fold the towel") == "fold_spread"


def test_prompt_is_outcome_neutral() -> None:
    prompt = task_prompt("container_transfer").lower()
    assert "failure" not in prompt
    assert "failed" not in prompt
    assert "container transfer" in prompt
