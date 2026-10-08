# Contributing

Keep changes scoped to a research question. Do not mix training-data changes, additional losses, model changes, and decoding changes into an unexplained comparison.

```bash
pip install -e '.[test]'
pytest tests/test_public_api.py tests/test_repository.py
python tools/check_repository.py
```

Use `src/crossinterest/` for importable code, `tests/` for regressions, `configs/` for portable configuration, and `docs/experiments/` for research documentation. Dataset and checkpoint locations belong in configuration, not user-specific absolute paths. Use lowercase snake-case experiment directories and descriptive filenames such as `report.md` and `results.md`.

Preserve the original prediction outputs and frozen contracts outside Git. A successful execution audit does not imply an improvement in model performance. Report data coverage, strict output validity, weight identity, and independent metric recomputation before interpreting results.

Do not commit credentials, Azure account details, user identifiers, downloaded datasets, model weights, local environments, or machine-specific job receipts. Record the model and dataset licenses separately from the project's own licensing decision.