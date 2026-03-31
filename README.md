# DSC 190/291: Machine Learning Competitions (SP26)

## Competitions

- `enhancer`
- `cashflow`
- `birdclef`
- `liverrisk`

## Setup

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you don't have it:
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install dependencies:
```bash
uv sync
```

## Submitting

Each competition has a model template in `competitions/<name>/model.py`.
Subclass the model and implement `predict()` and `process_inputs()`.
Submit your model as a `.py` file — we run it on a withheld test set to score.

See `docs/` for detailed instructions.
