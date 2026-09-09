from lapo_value_model.common import optimizer_schedule


def test_full_large_batch_step_budgets() -> None:
    teacher = optimizer_schedule(419_344, 2, 112, 1, 20)
    assert teacher["global_batch_size"] == 224
    assert teacher["optimizer_steps_per_epoch"] == 1_873
    assert teacher["total_optimizer_steps"] == 37_460

    student = optimizer_schedule(419_344, 2, 160, 1, 3)
    assert student["global_batch_size"] == 320
    assert student["optimizer_steps_per_epoch"] == 1_311
    assert student["total_optimizer_steps"] == 3_933

    screen = optimizer_schedule(419_344, 2, 160, 1, 3, max_optimizer_steps=400)
    assert screen["epochs"] == 1
    assert screen["micro_steps_per_epoch"] == 400
    assert screen["total_optimizer_steps"] == 400
