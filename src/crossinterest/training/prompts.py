"""Candidate-free Amazon SFT prompts aligned with the frozen v3 history views."""

from __future__ import annotations

from crossinterest.evaluation.build_views import HISTORY_INTRODUCTIONS, INPUT_TYPES, OUTPUT_INSTRUCTION, PREDICTION_INSTRUCTIONS, render_history

from crossinterest.training.label_space import LABEL_MODES, output_label
from crossinterest.training.label_space import INTEREST_ID_PATTERN


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
        "Predict the Interest ID for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should be a new topic that is not reflected in my product history."
    ),
    "interest": (
        "Predict the Interest ID for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should not appear in the history sequence above."
    ),
    "behavior_interest": (
        "Predict the Interest ID for the one new topic category that I am "
        "most likely to be interested in next. The category represented by the "
        "identifier should be a new topic that does not appear among the topic "
        "categories in my product history."
    ),
}
INTEREST_ID_OUTPUT_INSTRUCTION = (
    "Respond with only the single-token Interest ID in the exact format "
    "<interest_NNN>. Do not output the category name, spaces, or any explanation."
)
REASONING_INSTRUCTION = (
    "Analyze which observed evidence could support one plausible transition to the "
    "next new topic category. Briefly state the observed evidence, transition "
    "mechanism, and category role. Do not output an Interest ID."
)
REASONING_PREDICTION_INSTRUCTION = (
    "Based on the analysis above, predict the Interest ID for the one new topic "
    "category that I am most likely to be interested in next. "
    f"{INTEREST_ID_OUTPUT_INSTRUCTION}"
)
INTEREST_ID_CHOICE_OUTPUT_INSTRUCTION = (
    "Respond with only the single-token Interest ID assigned to the selected category, "
    "in the exact format <interest_NNN>. Do not output the category name, "
    "spaces, or any explanation."
)
INTEREST_ID_CATEGORY_TRAINING_INSTRUCTIONS = {
    input_type: instruction.replace(
        "Predict the Interest ID for the one new topic category",
        "Predict the Interest ID and exact category name for the one new topic category",
    )
    for input_type, instruction in INTEREST_ID_TRAINING_INSTRUCTIONS.items()
}
INTEREST_ID_CATEGORY_OUTPUT_INSTRUCTION = (
    "Respond with only the Interest ID followed by the exact category name in the "
    "format [<interest_NNN>] Parent Category > Child Category. Include the square "
    "brackets and one space after the closing bracket. Do not output any explanation."
)
INTEREST_ID_CATEGORY_CHOICE_OUTPUT_INSTRUCTION = (
    "Respond with only the Interest ID assigned to the selected category followed by "
    "that exact category name, in the format [<interest_NNN>] Parent Category > Child "
    "Category. Include the square brackets and one space after the closing bracket. "
    "Do not output the candidate number or any explanation."
)


def render_history_summary(row: dict) -> str:
    summary = row.get("history_summary")
    if summary is None:
        return ""
    if not isinstance(summary, str) or not summary.strip():
        raise ValueError("history_summary must be a nonempty string when present")
    return (
        "\n\nA separate model compressed only the observed history above into the "
        "following grounded user-state summary. Treat it as supporting context, not "
        "as a prediction:\n\n"
        f"{summary.strip()}"
    )


def render_training_prompt(row: dict, input_type: str, label_mode: str = "category") -> str:
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input type: {input_type}")
    if label_mode not in LABEL_MODES:
        raise ValueError(f"unsupported label mode: {label_mode}")
    instructions = {
        "category": TRAINING_INSTRUCTIONS,
        "interest_id": INTEREST_ID_TRAINING_INSTRUCTIONS,
        "interest_id_category": INTEREST_ID_CATEGORY_TRAINING_INSTRUCTIONS,
    }[label_mode]
    output_instruction = {
        "category": TRAINING_OUTPUT_INSTRUCTION,
        "interest_id": INTEREST_ID_OUTPUT_INSTRUCTION,
        "interest_id_category": INTEREST_ID_CATEGORY_OUTPUT_INSTRUCTION,
    }[label_mode]
    prompt = (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}"
        f"{render_history_summary(row)}\n\n"
        f"{instructions[input_type]}\n\n"
        f"{output_instruction}"
    )
    if "Candidates:" in prompt:
        raise AssertionError("SFT prompt must not contain a candidate list")
    return prompt


def render_reasoning_prompt(row: dict, input_type: str) -> str:
    if input_type not in INPUT_TYPES:
        raise ValueError(f"unsupported input type: {input_type}")
    prompt = (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}"
        f"{render_history_summary(row)}\n\n"
        f"{REASONING_INSTRUCTION}"
    )
    if "Candidates:" in prompt:
        raise AssertionError("reasoning prompt must not contain a candidate list")
    return prompt


def validate_target_reasoning(value: object, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string when present")
    if INTEREST_ID_PATTERN.search(value):
        raise ValueError(f"{field} must not contain an Interest ID")
    return value.strip()


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
    output_instruction = {
        "category": OUTPUT_INSTRUCTION,
        "interest_id": INTEREST_ID_CHOICE_OUTPUT_INSTRUCTION,
        "interest_id_category": INTEREST_ID_CATEGORY_CHOICE_OUTPUT_INSTRUCTION,
    }[label_mode]
    return (
        f"{HISTORY_INTRODUCTIONS[input_type]}\n\n"
        f"{render_history(row, input_type)}"
        f"{render_history_summary(row)}\n\n"
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
    target_reasoning = row.get("target_reasoning")
    target_rationale = row.get("target_rationale")
    if target_reasoning is not None and target_rationale is not None:
        raise ValueError("target_reasoning and target_rationale are mutually exclusive")
    if target_reasoning is not None:
        if label_mode != "interest_id":
            raise ValueError("target_reasoning requires single-token interest_id mode")
        return [
            {
                "role": "user",
                "content": render_reasoning_prompt(row, input_type),
            },
            {
                "role": "assistant",
                "content": validate_target_reasoning(
                    target_reasoning,
                    "target_reasoning",
                ),
            },
            {
                "role": "user",
                "content": REASONING_PREDICTION_INSTRUCTION,
            },
            {
                "role": "assistant",
                "content": output_label(target, label_mode, mapping),
            },
        ]
    messages = [
        {
            "role": "user",
            "content": render_training_prompt(row, input_type, label_mode),
        },
        {"role": "assistant", "content": output_label(target, label_mode, mapping)},
    ]
    if target_rationale is None:
        return messages
    if label_mode != "interest_id":
        raise ValueError("target_rationale requires single-token interest_id mode")
    messages.extend(
        [
            {
                "role": "user",
                "content": (
                    "Briefly explain which observed evidence and plausible transition "
                    "support that new category. Do not repeat its Interest ID."
                ),
            },
            {
                "role": "assistant",
                "content": validate_target_reasoning(
                    target_rationale,
                    "target_rationale",
                ),
            },
        ]
    )
    return messages