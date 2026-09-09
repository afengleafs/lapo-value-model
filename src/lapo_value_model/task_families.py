from __future__ import annotations

import re


TASK_FAMILIES = (
    "container_transfer",
    "reposition",
    "lid",
    "hang",
    "fold_spread",
    "hinged_open_close",
    "slidable_open_close",
    "button",
    "pour",
    "clean",
    "twist",
    "tool_use",
    "stir",
    "curtain",
    "bagging",
    "multi_step",
    "other",
)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def classify_task(raw: str | None) -> str:
    """Map heterogeneous DROID instructions to a shared outcome-neutral family."""
    text = " ".join((raw or "").strip().lower().split())
    first = text.split("|")[0]
    if not text or "no action" in text:
        return "other"
    if _has(text, r"two tasks|three tasks|multiple steps|consecutively|then reset|anything you like"):
        return "multi_step"
    if _has(text, r"curtain|shade|blinds"):
        return "curtain"
    if _has(text, r"stir|mix .* (bowl|pot|cup)"):
        return "stir"
    if _has(text, r"pour|empty .* (cup|bowl|container)|granular"):
        return "pour"
    if _has(text, r"clean|wipe|erase|scrub"):
        return "clean"
    if _has(text, r"button|switch|press|push the .*key"):
        return "button"
    if _has(text, r"faucet|tap|knob|twist|rotate .* (dial|handle)|turn (on|off) the (tap|faucet|stove)"):
        return "twist"
    if _has(text, r"fold|unfold|spread|clump|cloth|towel") and not _has(text, r"hang|hook|chair"):
        return "fold_spread"
    if _has(text, r"hang|unhang|hook|over (the )?chair"):
        return "hang"
    if _has(text, r"bag|unbag|zip|jacket"):
        return "bagging"
    if _has(text, r"lid|cap|cover"):
        return "lid"
    if _has(text, r"drawer|sliding|slidable|toaster") and _has(text, r"open|close|pull|push"):
        return "slidable_open_close"
    if _has(text, r"door|microwave|oven|book|laptop|dryer|toilet|hinged") and _has(text, r"open|close|shut"):
        return "hinged_open_close"
    if _has(text, r"spoon|fork|spatula|scoop|pick up something|use object to pick"):
        return "tool_use"
    if _has(text, r"into|inside|in the|out of|remove .* from|take .* out|container|drawer|hamper|trash|bin|bowl|cup|pot|box|sink|cab"):
        return "container_transfer"
    if _has(first, r"move|relocat|reposition|flip|pick and place|stack|place|put|grasp|shift|slide"):
        return "reposition"
    return "other"


def task_prompt(family: str) -> str:
    readable = family.replace("_", " ")
    return f"Robot manipulation task family: {readable}. Estimate current progress toward successful completion."

