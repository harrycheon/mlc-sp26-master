# Ramp-Up Assignment 2: Team Repo and Improved Model

**Due:** Sunday, April 12, 2026 at 11:59 PM
**Points:** 5
**Submission:** Push to your team GitHub repo + submit the Google Form

---

## Overview

In this assignment your team will set up a shared GitHub repository, choose a competition, and submit an improved model along with a report analyzing its performance.

## Prerequisites

- Completed Ramp-Up Assignment 1 (individual baseline model working)
- You should already be in a team

## Step 1: Create your team GitHub repository

Create a **private** GitHub repository for your team:

1. Go to [github.com/new](https://github.com/new)
2. Name your repo: `mlc-sp26-<teamname>` (e.g., `mlc-sp26-alpaca`)
3. Set visibility to **Private**
4. Create the repository
5. Add all team members as collaborators
6. Add the instructors as collaborators: go to **Settings > Collaborators** and add `ustunb` and `harrycheon`

## Step 2: Set up your project with uv

Install [uv](https://docs.astral.sh/uv/) if you haven't already:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

In your repo, initialize a project and pin the Python version:

```bash
cd mlc-sp26-<teamname>
uv init
uv python pin 3.11
```
*Note: You can use any python version*

Add dependencies as needed:

```bash
uv add numpy pandas scikit-learn
```

## Step 3: Clone the course repository

Clone the course repo to get the competition templates and data:

```bash
git clone <COURSE_REPO_URL>
```

## Step 4: Pick a competition and copy the template

As a team, choose one competition to work on. This will be your team's competition for the rest of the quarter.

| Competition | Template class | Metric |
|---|---|---|
| Enhancer | `EnhancerModel` | Mean AUPRC |
| Cashflow | `CashflowModel` | Min group ROC-AUC |
| BirdCLEF | `BirdclefModel` | Macro-avg ROC-AUC |
| LiverRisk | `LiverriskModel` | Weighted C-index |

Copy your chosen competition's directory and the base model template into your team repo:

```bash
cp -r dsc-mlc-sp26/competitions/<your-competition> ~/mlc-sp26-<teamname>/
cp -r dsc-mlc-sp26/src ~/mlc-sp26-<teamname>/
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
4. Your data directory should have this structure:
   ```
   birdclef/data/
   ├── train_audio/
   ├── test_soundscapes/
   └── train_metadata.csv
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

Do NOT commit downloaded data files to your repository — add them to `.gitignore`.

## Step 5: Implement an improved model

Create a file named `ramp2_submission.py` in your repo that:

1. **Subclasses the competition template** (e.g., `class MyModel(EnhancerModel)`)
2. **Implements `fit()`, `predict()`, and `process_inputs()`**
3. **Passes `__check_rep__`** — instantiating your model should not raise an error
4. **Achieves a higher `compute_score` than a simple baseline** (e.g., better than a default logistic regression or decision tree)

Your script must accept a `--data-dir` argument pointing to the competition data directory, and print the score when run directly. Add a `__main__` block that loads data, trains the model, generates predictions, and prints the result in this exact format:

```
Score: 0.6234
```

For example:

```python
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, required=True)
    args = parser.parse_args()

    model = MyModel(data_dir=args.data_dir)
    X, y = model.load_test_case()
    model.fit(X, y)
    predictions = model.predict(X)
    score = compute_score(y, predictions)
    print(f"Score: {score:.4f}")
```

## Step 6: Verify your model

Run your submission and confirm it prints the score:

```bash
uv run python ramp2_submission.py --data-dir <your-competition>/data
```

You should see output like:

```
Score: 0.6234
```

## Step 7: Write a performance report

Write a report (PDF, 2 pages max) analyzing your model's performance. Commit it to your repo as `ramp2_report.pdf`.

Your report should include:

1. **Overall performance** — Your `compute_score` result and the approach you used
2. **Subgroup analysis** — Break down performance by relevant subgroups for your competition:
   - *Enhancer:* per-chromosome AUPRC
   - *Cashflow:* per-group ROC-AUC
   - *BirdCLEF:* per-species ROC-AUC (or a representative subset)
   - *LiverRisk:* C-index for hepatic events vs. death separately
3. **Additional analysis** — At least one additional analysis component that your team came up with (e.g., error analysis, feature importance, comparison of model architectures, performance by data subset, calibration analysis)
4. **What you would try next** — Brief description of what you would improve given more time

## Step 8: Push to GitHub

Push your model and report to your team repo (`mlc-sp26-<teamname>`). Your repo should contain at minimum:

```
ramp2_submission.py                # your improved model
ramp2_report.pdf                   # performance report
pyproject.toml                     # project config with dependencies
uv.lock                            # locked dependency versions
src/                               # base model template (from course repo)
<competition>/model.py             # the competition template
<competition>/data/                # competition data files
```

## Submission

1. **GitHub:** Push your model and report to your `mlc-sp26-<teamname>` repo (make sure `ustunb`, `harrycheon`, and all team members are added as collaborators)
2. **Google Form:** Submit the [Competition Entry Form](https://docs.google.com/forms/d/e/1FAIpQLSda0RcEief_n28dryKcJEghcN64_1xXU09TVmuKAJ0KjpIgSg/viewform) — include a link to `ramp2_report.pdf` in your GitHub repo

We will clone your repo, run `uv run python ramp2_submission.py --data-dir <competition>/data`, and verify that the printed score matches what you reported.

## Grading

This assignment is worth **5 points**. We will evaluate your team repo setup, model submission, and performance report.

## Troubleshooting

**My model passes `__check_rep__` but my score didn't improve:**
Try different feature engineering, model architectures, or hyperparameter tuning. Look at where your model performs worst in the subgroup analysis for clues.

**Team members can't push to the repo:**
Make sure all team members are added as collaborators with write access in **Settings > Collaborators**.

**Data files are too large for GitHub:**
Add data directories to `.gitignore`. Each team member should download the data locally.
