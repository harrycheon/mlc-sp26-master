# Ramp-Up Assignment 1: Baseline Model

**Due:** Sunday, April 5, 2026 at 11:59 PM
**Submission:** Push baseline model to your GitHub repo + submit the Google Form

---

## Overview

In this assignment you will create a GitHub repository for your course work, set up a Python project with reproducible dependencies, and build a working baseline model for your chosen competition.

## Step 1: Create your GitHub repository

Create a **private** GitHub repository for your course work:

1. Go to [github.com/new](https://github.com/new)
2. Name your repo: `mlc-sp26-<your-ucsd-username>` (e.g., `mlc-sp26-jdoe`)
3. Set visibility to **Private**
4. Create the repository
5. Add the instructors as collaborators: go to **Settings > Collaborators** and add `ustunb` and `harrycheon`

## Step 2: Set up your project with uv

We use [uv](https://docs.astral.sh/uv/) to manage Python versions and dependencies. Install it if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

In your repo, initialize a project:

```bash
cd mlc-sp26-<username>
uv init
```

This creates a `pyproject.toml` where you can specify your Python version and dependencies. Add packages as needed (e.g., for a scikit-learn baseline):

```bash
uv add numpy pandas scikit-learn
```

Running `uv add` updates `pyproject.toml` and creates a `uv.lock` that pins your exact dependency versions so we can reproduce your results.

## Step 3: Clone the course repository

Clone the course repo to get the competition templates and data:

```bash
git clone https://github.com/harrycheon/mlc-sp26-master.git
```

## Step 4: Pick a competition and copy the template

| Competition | Template class | Metric |
|---|---|---|
| Enhancer | `EnhancerModel` | Mean AUPRC |
| Cashflow | `CashflowModel` | Mean ROC-AUC across loan types |
| BirdCLEF | `BirdclefModel` | Macro-avg ROC-AUC |
| LiverRisk | `LiverriskModel` | Weighted C-index |

Copy your chosen competition's directory and the base model template into your personal repo:

```bash
cp -r dsc-mlc-sp26/competitions/<your-competition> ~/mlc-sp26-<username>/
cp -r dsc-mlc-sp26/src ~/mlc-sp26-<username>/
```

**BirdCLEF** and **LiverRisk** require downloading data from external sources. If you chose one of these competitions, follow the steps below before proceeding.

**BirdCLEF — download from Kaggle:**

1. Install the Kaggle CLI: `uv add kaggle`
2. Set up your API credentials: https://www.kaggle.com/docs/api
3. Download and extract the data:
   ```bash
   kaggle competitions download -c birdclef-2026 -p birdclef/data/
   unzip birdclef/data/birdclef-2026.zip -d birdclef/data/
   ```
4. Note: Only a small subset of `train_soundscapes/` files have expert labels (in `train_soundscapes_labels.csv`). Scoring evaluates only on these labeled segments.
5. Your data directory should have this structure:
   ```
   birdclef/data/
   ├── train_audio/
   ├── train_soundscapes/
   ├── test_soundscapes/
   ├── train_soundscapes_labels.csv
   ├── taxonomy.csv
   ├── train.csv
   └── sample_submission.csv
   ```

**LiverRisk — download from Trustii:**

1. Register on Trustii: https://app.trustii.io
2. Join the ANNITIA challenge: https://app.trustii.io/datasets/1551
3. Download the data files from the challenge page
4. Place them in `liverrisk/data/` with these exact names:
   ```
   liverrisk/data/
   ├── train.csv
   ├── test.csv
   └── dictionary.csv
   ```
5. Install `scikit-survival`: `uv add scikit-survival`

Do NOT commit downloaded data files to your repository — add them to `.gitignore`.

## Step 5: Implement a baseline model

Create a file named `ramp1_submission.py` in your repo that:

1. **Subclasses the competition template** (e.g., `class MyModel(EnhancerModel)`)
2. **Implements `fit()`, `predict()`, and `process_inputs()`**
3. **Passes `__check_rep__`** — instantiating your model should not raise an error

Your model does not need to be accurate — a simple logistic regression or decision tree with basic features is fine. The goal is a working pipeline, not a high score. Your script should run relatively quickly (~1 min).

Note: `score_model` trains and evaluates on the same data. The reported score reflects pipeline correctness, not generalization. We will use held-out data for final evaluation.

Your script must accept a `--data-dir` argument pointing to the competition data directory, and print the score when run directly. Add a `__main__` block that loads data, trains the model, generates predictions, and prints the result in this exact format:

```
Score: 0.6234
```

For example:

```python
import argparse
from enhancer.model import EnhancerModel, score_model

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    model = MyModel()
    score = score_model(model, args.data_dir)
    print(f"Score: {score:.4f}")
```

## Step 6: Verify your model

Run your submission and confirm it prints the score:

```bash
uv run python ramp1_submission.py --data-dir <your-competition>/data
```

You should see output like:

```
Score: 0.6234
```

## Step 7: Push to GitHub

Push your baseline model to your private repo (`mlc-sp26-<username>`). Your repo should contain at minimum:

```
ramp1_submission.py                # your baseline model
pyproject.toml                     # project config with dependencies
uv.lock                            # locked dependency versions
src/                               # base model template (from course repo)
<competition>/model.py             # the competition template
<competition>/data/                # competition data files
```

## Submission

1. **GitHub:** Push your baseline model to your `mlc-sp26-<username>` repo (make sure `ustunb` and `harrycheon` are added as collaborators)
2. **Google Form:** Submit the [Ramp-Up Assignment 1 form](https://docs.google.com/forms/d/e/1FAIpQLSfVqsnPryWBclhKUWdaMYkBcsTsRSIgXEIlDWzN8li-Op4q4Q/viewform)

We will clone your repo, run `uv run python ramp1_submission.py --data-dir <competition>/data`, and verify that the printed score matches what you reported.
