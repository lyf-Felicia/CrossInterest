"""Paper-faithful K=2 cluster prompts adapted from short videos to Amazon."""

from __future__ import annotations


PROMPT_SCHEMA_VERSION = "google-interest-v2"
FULL_HISTORY_RANKING_PROMPT_SCHEMA_VERSION = "google-interest-full-history-ranking-v1"
OUTPUT_PREFIX = "New cluster: "


def cluster_terms(cluster: str) -> tuple[str, str]:
    parts = tuple(part.strip() for part in cluster.split(" > ", 1))
    if len(parts) != 2 or not all(parts):
        raise ValueError(f"malformed Amazon L2 > L3 cluster: {cluster!r}")
    return parts


def cluster_description(cluster: str) -> str:
    return ", ".join(cluster_terms(cluster))


def cluster_output(cluster: str) -> str:
    return OUTPUT_PREFIX + cluster_description(cluster)


def select_recent_unique_clusters(
    history: list[str],
    supported_clusters: set[str],
    count: int = 2,
) -> list[str] | None:
    if count < 1:
        raise ValueError("count must be positive")
    selected_reversed: list[str] = []
    seen: set[str] = set()
    for cluster in reversed(history):
        if cluster not in supported_clusters or cluster in seen:
            continue
        selected_reversed.append(cluster)
        seen.add(cluster)
        if len(selected_reversed) == count:
            return list(reversed(selected_reversed))
    return None


def render_generation_prompt(cluster_pair: list[str] | tuple[str, str]) -> str:
    if len(cluster_pair) != 2:
        raise ValueError("the paper-faithful prompt requires exactly two clusters")
    first, second = (cluster_description(cluster) for cluster in cluster_pair)
    return (
        "The product categories I interacted with most recently are in the following "
        "clusters:\n"
        f"Cluster 1: {{{first}}}\n"
        f"Cluster 2: {{{second}}}.\n"
        "Each cluster is described with salient phrases or entity names. With less "
        "than 30 words, generate a new and different product category cluster I will "
        "interact with next with highly specific salient phrases or entity names, with\n"
        'a prefix that says "New cluster: "'
    )


def render_full_history_generation_prompt(
    historical_titles: list[str],
    historical_interests: list[str],
) -> str:
    if not historical_interests:
        raise ValueError("full-history ranking requires at least one cluster")
    if len(historical_titles) != len(historical_interests):
        raise ValueError("title and cluster histories are not aligned")
    history = "\n".join(
        f"{index}. [{cluster}] {title}"
        for index, (title, cluster) in enumerate(
            zip(historical_titles, historical_interests, strict=True),
            1,
        )
    )
    return (
        "The titles and corresponding topic categories of the products I recently "
        "rated positively, in chronological order, are listed below. Each item is "
        'formatted as "[Topic category] Product title":\n\n'
        f"{history}\n\n"
        "With less than 30 words, generate a new and different product category cluster I will "
        "interact with next with highly specific salient phrases or entity names, with\n"
        'a prefix that says "New cluster: "'
    )


def render_classification_prompt(
    cluster_pair: list[str] | tuple[str, str],
    candidates: list[str],
) -> str:
    if len(candidates) != 10 or len(set(candidates)) != 10:
        raise ValueError("classification requires exactly 10 unique candidates")
    choices = "\n".join(
        f"{index}. {{{cluster_description(candidate)}}}"
        for index, candidate in enumerate(candidates, 1)
    )
    return (
        f"{render_generation_prompt(cluster_pair)}\n\n"
        "Choose the one most likely new cluster from the following candidate clusters:\n"
        f"{choices}\n\n"
        'Respond with only the selected cluster using the prefix "New cluster: ".'
    )


def render_training_messages(row: dict) -> list[dict[str, str]]:
    pair = list(row["cluster_pair"])
    target = row["target_cluster"]
    if target in pair:
        raise ValueError("the novel target cluster appears in the K=2 context")
    return [
        {"role": "user", "content": render_generation_prompt(pair)},
        {"role": "assistant", "content": cluster_output(target)},
    ]