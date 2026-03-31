"""Enhancer (gene regulation) model and scoring template.

Competition: Predict whether a candidate enhancer regulates a target gene.
Input: CSV features (genomic features for enhancer-gene pairs)
Output: Binary predictions (True/False for regulation)
Metric: Mean AUPRC across chromosomes
"""
import numpy as np
import pandas as pd
from abc import abstractmethod
from pathlib import Path
from sklearn.metrics import average_precision_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.base import ScorableModelTemplate

DATA = Path(__file__).resolve().parent / "data"


class EnhancerModel(ScorableModelTemplate):
    """Abstract model for the enhancer regulation competition.

    Students subclass this and implement predict() and process_inputs().
    """

    def load_test_case(self):
        """Load a sample from training data for local validation.

        Returns:
            tuple: (X_sample DataFrame, y_sample Series)
        """
        X = pd.read_csv(str(DATA / "X.csv"))
        y = pd.read_csv(str(DATA / "y.csv")).squeeze()
        # Use a small sample for quick validation
        sample_idx = X.sample(n=min(100, len(X)), random_state=42).index
        return X.loc[sample_idx], y.loc[sample_idx]

    def __check_rep__(self):
        """Validate fit and predict on a training data sample."""
        X_sample, y_sample = self.load_test_case()

        # Fit
        try:
            self.fit(X_sample, y_sample)
        except Exception as e:
            raise ValueError(f"fit function does not work: {e}")

        # Predict
        try:
            predicted = self.predict(X_sample)
        except Exception as e:
            raise ValueError(f"predict function does not work: {e}")

        # Score
        try:
            compute_score(y_sample, predicted)
        except Exception as e:
            raise ValueError(f"prediction could not be scored: {e}")

        # Checks
        assert isinstance(predicted, (np.ndarray, list, pd.Series)), \
            "output should be array-like (np.ndarray, list, or pd.Series)"
        assert len(predicted) == len(y_sample), \
            f"expected {len(y_sample)} predictions, got {len(predicted)}"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series):
        """Train the model on the given data.

        Args:
            X: Feature DataFrame.
            y: Binary labels (0 or 1).
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict regulation probability for each enhancer-gene pair.

        Args:
            X: Feature DataFrame with the same columns as X_train.

        Returns:
            Array of predicted probabilities (floats between 0 and 1).
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, X: pd.DataFrame):
        """Preprocess features for the model.

        Args:
            X: Raw feature DataFrame.

        Returns:
            Processed features suitable for the model.
        """
        raise NotImplementedError()


def compute_score(y_true, y_pred) -> float:
    """Compute mean AUPRC score for enhancer predictions.

    Args:
        y_true: Ground truth binary labels.
        y_pred: Predicted probabilities.

    Returns:
        Area under the Precision-Recall Curve.
    """
    return average_precision_score(y_true, y_pred)


def score_model(model: EnhancerModel, data_dir: str) -> float:
    """Score a model by mean AUPRC across chromosomes.

    Trains the model on the full dataset, predicts on it, then computes
    AUPRC per chromosome and returns the mean.

    Args:
        model: An instantiated EnhancerModel subclass.
        data_dir: Path to a data directory containing X.csv and y.csv.

    Returns:
        Mean AUPRC across chromosomes.
    """
    data_dir = Path(data_dir)

    X = pd.read_csv(str(data_dir / "X.csv"))
    y = pd.read_csv(str(data_dir / "y.csv")).squeeze().astype(int)

    model.fit(X, y)
    predictions = model.predict(X)

    chroms = sorted(X["chr"].unique())
    auprc_scores = []
    for chrom in chroms:
        mask = X["chr"] == chrom
        if y[mask].sum() == 0:
            continue
        auprc_scores.append(compute_score(y[mask], predictions[mask]))

    return np.mean(auprc_scores)
