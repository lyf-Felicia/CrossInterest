"""Shared prompt contract for MIND behavior and interest views."""

from __future__ import annotations


VIEWS = ("behavior", "interest")
SYSTEM_PROMPT = (
    "Predict the user's next novel news topic from their chronological reading history. "
    "A novel topic is a subcategory that does not appear in the displayed history. "
    'Reply with exactly "New topic: <subcategory>" and no other text.'
)
HISTORY_INTRODUCTIONS = {
    "behavior": (
        "The titles of news articles the user recently read, in chronological order, "
        "are listed below:"
    ),
    "interest": (
        "The topic subcategories associated with news articles the user recently read, "
        "in chronological order, are listed below:"
    ),
}
PREDICTION_INSTRUCTIONS = {
    "behavior": (
        "Predict the one topic subcategory the user is most likely to read next. "
        "It must represent a new topic not reflected in the article-title history above."
    ),
    "interest": (
        "Predict the one topic subcategory the user is most likely to read next. "
        "It must not appear in the history sequence above."
    ),
}
OUTPUT_INSTRUCTION = 'Reply with exactly "New topic: <subcategory>" and no other text.'


def render_history(history: list[dict], view: str) -> str:
    if view not in VIEWS:
        raise ValueError(f"unsupported view: {view}")
    if view == "behavior":
        values = [item["title"] for item in history]
    else:
        values = [item["subcategory"] for item in history]
    return "\n".join(f"{index}. {value}" for index, value in enumerate(values, 1))


def render_user_prompt(
    history: list[dict], view: str, candidates: list[str] | None = None
) -> str:
    sections = [
        HISTORY_INTRODUCTIONS[view],
        render_history(history, view),
        PREDICTION_INSTRUCTIONS[view],
    ]
    if candidates is not None:
        if len(candidates) != 10 or len(set(candidates)) != 10:
            raise ValueError("choice evaluation requires exactly 10 unique candidates")
        sections.extend(
            [
                "Candidate topic subcategories:",
                "\n".join(
                    f"{index}. {label}" for index, label in enumerate(candidates, 1)
                ),
                "Select exactly one topic subcategory from the candidate list.",
            ]
        )
    sections.append(OUTPUT_INSTRUCTION)
    return "\n\n".join(sections)


def render_messages(
    history: list[dict],
    view: str,
    target: str | None = None,
    candidates: list[str] | None = None,
) -> list[dict[str, str]]:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": render_user_prompt(history, view, candidates)},
    ]
    if target is not None:
        messages.append({"role": "assistant", "content": f"New topic: {target}"})
    return messages
