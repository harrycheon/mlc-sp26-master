"""BirdCLEF 2026 model and scoring template.

Competition: Identify bird species from soundscape audio recordings.
URL: https://www.kaggle.com/competitions/birdclef-2026
Input: Directory of OGG audio files (32kHz soundscapes from Pantanal wetlands)
Output: DataFrame with row_id + per-species predicted probabilities (0-1)
Metric: Macro-averaged ROC-AUC across species
"""
import numpy as np
import pandas as pd
from abc import abstractmethod
from pathlib import Path
from sklearn.metrics import roc_auc_score

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))
from src.base import ScorableModelTemplate

DATA = Path(__file__).resolve().parent / "data"


def _start_to_seconds(s: str) -> int:
    """Convert 'H:MM:SS' timestamp to total seconds."""
    h, m, sec = s.split(":")
    return int(h) * 3600 + int(m) * 60 + int(sec)


def build_ground_truth(labels: pd.DataFrame, taxonomy: pd.DataFrame) -> pd.DataFrame:
    """Build a ground truth DataFrame from labels and taxonomy.

    Converts train_soundscapes_labels.csv (H:MM:SS timestamps) into the
    submission format (filename_seconds row IDs) used by sample_submission.csv.

    Args:
        labels: DataFrame with 'filename', 'start', and 'primary_label' columns
            (from train_soundscapes_labels.csv).
        taxonomy: DataFrame with 'primary_label' column listing all species.

    Returns:
        DataFrame with 'row_id' column (format: filename_seconds) and one
        binary column per species.
    """
    species_cols = taxonomy["primary_label"].astype(str).tolist()
    labels = labels.copy()
    labels["row_id"] = (
        labels["filename"].str.replace(".ogg", "", regex=False)
        + "_"
        + labels["end"].apply(_start_to_seconds).astype(str)
    )
    split_labels = labels["primary_label"].apply(lambda x: set(str(x).split(";")))
    binary = {col: split_labels.apply(lambda x: 1 if col in x else 0).values
              for col in species_cols}
    return pd.DataFrame({"row_id": labels["row_id"].values, **binary})


class BirdclefModel(ScorableModelTemplate):
    """Abstract model for the BirdCLEF 2026 competition.

    Students subclass this and implement fit(), predict(), and process_inputs().

    Note: Only a small subset of train_soundscapes/ have expert labels
    (in train_soundscapes_labels.csv). Scoring evaluates only on labeled
    segments. Your predict() may return predictions for all segments —
    only matching row_ids will be scored.
    """

    def load_test_case(self):
        """Load a small sample of labeled train soundscapes for validation.

        Returns:
            tuple: (sample_files, y_true) where sample_files is a list of
                filenames and y_true is a DataFrame in submission format
                (row_id + binary species columns).
        """
        labels = pd.read_csv(str(DATA / "train_soundscapes_labels.csv"))
        taxonomy = pd.read_csv(str(DATA / "taxonomy.csv"))

        # Pick first 2 soundscape files (small, fast)
        sample_files = sorted(labels["filename"].unique())[:2]
        sample = labels[labels["filename"].isin(sample_files)]
        y_true = build_ground_truth(sample, taxonomy)

        return list(sample_files), y_true

    def __check_rep__(self):
        """Validate predict() on a small soundscape sample."""
        if not (DATA / "train_soundscapes_labels.csv").exists():
            return  # Skip if data not downloaded

        import tempfile, shutil
        sample_files, y_true = self.load_test_case()

        tmpdir = tempfile.mkdtemp()
        try:
            for f in sample_files:
                shutil.copy2(str(DATA / "train_soundscapes" / f), tmpdir)

            try:
                self.fit(tmpdir, y_true)
            except Exception as e:
                raise ValueError(f"fit() failed: {e}")

            try:
                y_pred = self.predict(tmpdir)
            except Exception as e:
                raise ValueError(f"predict() failed: {e}")

            assert isinstance(y_pred, pd.DataFrame), \
                "predict() must return a DataFrame"
            assert "row_id" in y_pred.columns, \
                "predict() output must have a 'row_id' column"

            pred_species = [c for c in y_pred.columns if c != "row_id"]
            assert len(pred_species) > 0, \
                "predict() output must have species probability columns"
            for col in pred_species:
                vals = y_pred[col]
                assert vals.between(0, 1).all(), \
                    f"Species column '{col}' has values outside [0, 1]"
        finally:
            shutil.rmtree(tmpdir)

    @abstractmethod
    def fit(self, audio_dir: str, labels: pd.DataFrame):
        """Train the model on labeled soundscape data.

        Args:
            audio_dir: Path to directory containing OGG soundscape files.
            labels: DataFrame with 'row_id' and binary species columns
                (ground truth in submission format).
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, audio_dir: str) -> pd.DataFrame:
        """Predict bird species probabilities from soundscape audio.

        Args:
            audio_dir: Path to directory containing OGG soundscape files.

        Returns:
            DataFrame with 'row_id' column (format: filename_timeoffset)
            and one column per species containing predicted probabilities
            (floats between 0 and 1). This is a multi-label task — multiple
            species can be present in a single 5-second window.
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, audio_dir: str):
        """Preprocess audio files for the model.

        Args:
            audio_dir: Path to directory containing OGG soundscape files.

        Returns:
            Processed features suitable for the model (e.g., mel spectrograms).
        """
        raise NotImplementedError()


def compute_score(y_true: pd.DataFrame, y_pred: pd.DataFrame) -> float:
    """Compute macro-averaged ROC-AUC across species.

    Args:
        y_true: Ground truth DataFrame with row_id + binary species columns.
        y_pred: Predicted DataFrame with row_id + species probability columns.

    Returns:
        Macro-averaged ROC-AUC score (species with no positive labels are skipped).
    """
    species_cols = [c for c in y_true.columns if c != 'row_id']
    aucs = []
    for col in species_cols:
        if y_true[col].nunique() < 2:
            continue
        aucs.append(roc_auc_score(y_true[col], y_pred[col]))
    if not aucs:
        raise ValueError("No species with both positive and negative labels found")
    return np.mean(aucs)


def score_model(model: BirdclefModel, data_dir: str) -> float:
    """Score a model against the labeled train soundscapes.

    Args:
        model: An instantiated BirdclefModel subclass.
        data_dir: Path to a data directory containing train_soundscapes/,
            train_soundscapes_labels.csv, and taxonomy.csv.

    Returns:
        Macro-averaged ROC-AUC score.
    """
    data_dir = Path(data_dir)
    labels = pd.read_csv(str(data_dir / "train_soundscapes_labels.csv"))
    taxonomy = pd.read_csv(str(data_dir / "taxonomy.csv"))
    y_true = build_ground_truth(labels, taxonomy)

    # Fit and predict
    audio_dir = str(data_dir / "train_soundscapes")
    model.fit(audio_dir, y_true)
    y_pred = model.predict(audio_dir)

    species_cols = [c for c in y_true.columns if c != "row_id"]
    merged = y_true.merge(y_pred, on="row_id", suffixes=("_true", "_pred"))
    y_true_aligned = merged[["row_id"] + [f"{c}_true" for c in species_cols]].rename(
        columns={f"{c}_true": c for c in species_cols}
    )
    y_pred_aligned = merged[["row_id"] + [f"{c}_pred" for c in species_cols]].rename(
        columns={f"{c}_pred": c for c in species_cols}
    )

    return compute_score(y_true_aligned, y_pred_aligned)
