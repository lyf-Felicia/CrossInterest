# Prompt Protocol

All four training conditions use the same structured parquet rows and differ only in the runtime view renderer. The history is ordered from oldest to newest and capped at the most recent 50 interactions.

## Shared system message

Both views use this exact system message:

```text
Predict the user's next novel news topic from their chronological reading history. A novel topic is a subcategory that does not appear in the displayed history. Reply with exactly "New topic: <subcategory>" and no other text.
```

## Behavior view

The behavior view exposes news titles only. No category, subcategory, or candidate list is shown during training or ranking.

### Exact user message for training and ranking

```text
The titles of news articles the user recently read, in chronological order, are listed below:

1. <news title>
2. <news title>
...

Predict the one topic subcategory the user is most likely to read next. It must represent a new topic not reflected in the article-title history above.

Reply with exactly "New topic: <subcategory>" and no other text.
```

## Interest view

The interest view exposes historical subcategories only. No title, category, or candidate list is shown during training or ranking.

### Exact user message for training and ranking

```text
The topic subcategories associated with news articles the user recently read, in chronological order, are listed below:

1. <historical subcategory>
2. <historical subcategory>
...

Predict the one topic subcategory the user is most likely to read next. It must not appear in the history sequence above.

Reply with exactly "New topic: <subcategory>" and no other text.
```

### Assistant message for training

```text
New topic: <gold subcategory>
```

## 10-way choice evaluation

Only Macro Accuracy@10 receives candidates. The following exact block is inserted after the selected view's prediction instruction and before its final reply instruction:

```text

Candidate topic subcategories:
1. <candidate 1>
2. <candidate 2>
...
10. <candidate 10>

Select exactly one topic subcategory from the candidate list.
```

The ten candidates are unique and include the gold subcategory exactly once. Their order is deterministically rotated across examples.

## Candidate isolation

| Stage | Candidate list in prompt | Output space |
|---|---:|---|
| SFT training | No | Gold assistant answer |
| SuperLoss training | No | Gold assistant answer |
| Macro Accuracy@10 | Yes, exactly 10 | The 10 shown candidates |
| Macro Recall/nDCG@10,@20 | No | All 124 taxonomy labels |

Ranking uses constrained beam search over tokenized legal output strings. This constraint validates the output grammar; it does not place taxonomy labels in the prompt.
