"""Candidate-free Amazon SFT prompts aligned with the frozen v3 history views."""

from __future__ import annotations

from crossinterest.evaluation.build_views import HISTORY_INTRODUCTIONS, INPUT_TYPES, OUTPUT_INSTRUCTION, PREDICTION_INSTRUCTIONS, render_history

from crossinterest.legacy.training_99.label_space import LABEL_MODES, output_label


TRAINING_INSTRUCTIONS = {
    "behavior": (
        "Predict the one new topic category that I am most likely to be interested in "
        "next. The category should represent a new topic that is not reflected in my "
        "product history."
    ),
    "interest": (
        "Predict the one new topic category that I am most likely to be interested in "
        "next. The category should not appear in the history sequence above."
    ),
    "behavior_interest": (
        "Predict the one new topic category that I am most likely to be interested in "
        "next. The category should represent a new topic that does not appear among the "
        "topic categories in my product history."
    ),
}
TRAINING_OUTPUT_INSTRUCTION = "Respond with only the exact category name."
INTEREST_ID_TRAINING_INSTRUCTIONS = {
    "behavior": (
        "Predict the interest identifier for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should be a new topic that is not reflected in my product history."
    ),
    "interest": (
        "Predict the interest identifier for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should not appear in the history sequence above."
    ),
    "behavior_interest": (
        "Predict the interest identifier for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should be a new topic that does not appear among the topic "
        "categories in my product history."
    ),
}
INTEREST_ID_OUTPUT_INSTRUCTION = (
    "Respond with only the interest identifier in the exact format <interest_NNN>, "
    "where NNN is a three-digit number. Do not output the category name or any explanation."
)
INTEREST_ID_CHOICE_OUTPUT_INSTRUCTION = (
    "Respond with only the interest identifier assigned to the selected category, in "
    "the exact format <interest_NNN>. Do not output the category name or any explanation."
)


def render_training_prompt(row: dict, input_type: str, label_mode: str = "category") -> str:
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input type: {input_type}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label mode: {label_mode}")
    instructions = (
        TRAINING_INSTRUCTIONS
        if label_mode == "category"
        else INTEREST_ID_TRAINING_INSTRUCTIONS
    )
    output_instruction = (
        TRAINING_OUTPUT_INSTRUCTION
        if label_mode == "category"
        else INTEREST_ID_OUTPUT_INSTRUCTION
    )
    prompt = (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}\n\n"
        f"{instructions[input_type]}\n\n"
        f"{output_instruction}"
    )
    if "Candidates:" in prompt:
        raise AssertionError("SFT prompt must not contain a candidate list")
    return prompt


def render_classification_prompt(
    row: dict,
    candidates: list[str],
    input_type: str,
    label_mode: str = "category",
) -> str:
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input type: {input_type}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label mode: {label_mode}")
    candidate_text = "\n".join(
        f"{index}. {candidate}" for index, candidate in enumerate(candidates, 1)
    )
    output_instruction = (
        OUTPUT_INSTRUCTION
        if label_mode == "category"
        else INTEREST_ID_CHOICE_OUTPUT_INSTRUCTION
    )
    return (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}\n\n"
        f"{PREDICTION_INSTRUCTIONS[input_type]}\n\n"
        f"Candidates:\n\n{candidate_text}\n\n"
        f"{output_instruction}"
    )


def render_training_messages(
    row: dict,
    input_type: str,
    label_mode: str = "category",
    mapping: dict | None = None,
) -> list[dict[str, str]]:
    target = row["target_interest"]
    if target in set(row["full_seen_interests"]):
        raise ValueError(f"{row['id']}: target appears in complete history")
    return [
        {
            "role": "user",
            "content": render_training_prompt(row, input_type, label_mode),
        },
        {"role": "assistant", "content": output_label(target, label_mode, mapping)},
    ]