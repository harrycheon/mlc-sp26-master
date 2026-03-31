"""LiverRisk (ANNITIA) model and scoring template.

Competition: Predict major liver events or death in MASLD patients.
URL: https://app.trustii.io/datasets/1551
Input: train.csv (clinical data with outcomes), test.csv (clinical data without outcomes)
Output: DataFrame with trustii_id, risk_hepatic_event, risk_death
Metric: Weighted C-index: 0.3 * C-index(death) + 0.7 * C-index(hepatic)
"""
import numpy as np
import pandas as pd
from abc import abstractmethod
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.base import ScorableModelTemplate

DATA = Path(__file__).resolve().parent / "data"


class LiverriskModel(ScorableModelTemplate):
    """Abstract model for the LiverRisk (ANNITIA) competition.

    Students subclass this and implement fit(), predict(), and process_inputs().

    This is a survival analysis task. The outcome columns encode time-to-event
    data: an event indicator (0/1) and an age-at-occurrence. Patients without
    an event are "censored" — we know they were event-free up to a certain age
    but not what happens after. The metric is the C-index (concordance index),
    which measures how well predicted risk scores rank patients by event time.

    Requires scikit-survival: `uv add scikit-survival`
    Docs: https://scikit-survival.readthedocs.io

    Data layout:
        train.csv — 1253 patients with features + outcome columns:
            evenements_hepatiques_majeurs (0/1), evenements_hepatiques_age_occur,
            death (0/1), death_age_occur
        test.csv — 423 patients with features only (+ trustii_id column)
        sample_submission.csv — expected output format:
            trustii_id, risk_hepatic_event, risk_death
    """

    def load_test_case(self):
        """Load a sample from training data for local validation.

        Returns:
            tuple: (X_sample DataFrame, y_sample DataFrame) where y_sample
                has columns: evenements_hepatiques_majeurs,
                evenements_hepatiques_age_occur, death, death_age_occur.
        """
        train = pd.read_csv(str(DATA / "train.csv"))
        outcome_cols = [
            "evenements_hepatiques_majeurs",
            "evenements_hepatiques_age_occur",
            "death",
            "death_age_occur",
        ]
        feature_cols = [c for c in train.columns if c not in outcome_cols]
        sample_idx = train.sample(n=min(100, len(train)), random_state=42).index
        return train.loc[sample_idx, feature_cols], train.loc[sample_idx, outcome_cols]

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

        # Checks
        assert isinstance(predicted, pd.DataFrame), \
            "predict() should return a DataFrame"
        for col in ["risk_hepatic_event", "risk_death"]:
            assert col in predicted.columns, \
                f"predict() output missing required column: {col}"
        assert len(predicted) == len(X_sample), \
            f"expected {len(X_sample)} predictions, got {len(predicted)}"

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.DataFrame):
        """Train the model on the given data.

        Args:
            X: Feature DataFrame (all columns except outcome columns).
            y: Outcome DataFrame with columns: evenements_hepatiques_majeurs,
                evenements_hepatiques_age_occur, death, death_age_occur.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict risk scores for liver events and death.

        Args:
            X: Feature DataFrame with the same columns as training features.

        Returns:
            DataFrame with columns: risk_hepatic_event, risk_death.
            Higher scores indicate higher risk. One row per patient.
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, X: pd.DataFrame):
        """Preprocess longitudinal clinical and NIT data for the model.

        Args:
            X: Raw feature DataFrame.

        Returns:
            Processed features suitable for the model.
        """
        raise NotImplementedError()


def compute_score(
    y_true: pd.DataFrame,
    y_pred: pd.DataFrame,
) -> float:
    """Compute weighted C-index for liver event and death predictions.

    Score = 0.3 * C-index(death) + 0.7 * C-index(hepatic)

    Uses concordance_index_censored from scikit-survival.

    Args:
        y_true: DataFrame with columns: evenements_hepatiques_majeurs,
            evenements_hepatiques_age_occur, death, death_age_occur.
        y_pred: DataFrame with columns: risk_hepatic_event, risk_death.

    Returns:
        Weighted concordance index score.
    """
    from sksurv.metrics import concordance_index_censored

    c_hepatic = concordance_index_censored(
        y_true["evenements_hepatiques_majeurs"].astype(bool).values,
        y_true["evenements_hepatiques_age_occur"].fillna(
            y_true["evenements_hepatiques_age_occur"].max()
        ).values,
        y_pred["risk_hepatic_event"].values,
    )[0]

    c_death = concordance_index_censored(
        y_true["death"].astype(bool).values,
        y_true["death_age_occur"].fillna(
            y_true["death_age_occur"].max()
        ).values,
        y_pred["risk_death"].values,
    )[0]

    return 0.7 * c_hepatic + 0.3 * c_death


def score_model(model: LiverriskModel, data_dir: str) -> float:
    """Score a model against labeled data via cross-validation on train.csv.

    Since test.csv has no labels (scored on Trustii), local scoring
    uses the training data.

    Args:
        model: An instantiated LiverriskModel subclass.
        data_dir: Path to a data directory containing train.csv.

    Returns:
        Weighted C-index score on training data.
    """
    data_dir = Path(data_dir)

    train = pd.read_csv(str(data_dir / "train.csv"))
    outcome_cols = [
        "evenements_hepatiques_majeurs",
        "evenements_hepatiques_age_occur",
        "death",
        "death_age_occur",
    ]
    feature_cols = [c for c in train.columns if c not in outcome_cols]

    X = train[feature_cols]
    y_true = train[outcome_cols]

    model.fit(X, y_true)
    y_pred = model.predict(X)

    return compute_score(y_true, y_pred)
