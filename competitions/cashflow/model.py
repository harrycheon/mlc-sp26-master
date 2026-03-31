"""Cashflow model and scoring template.

Competition: Predict whether a consumer will experience financial distress.
Input: Consumer demographics (parquet) + transaction history (parquet)
Output: Predicted probabilities (float 0-1) per consumer
Metric: Mean ROC-AUC across loan types
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


class CashflowModel(ScorableModelTemplate):
    """Abstract model for the cashflow competition.

    Students subclass this and implement predict() and process_inputs().
    """

    def load_test_case(self):
        """Load test case data.

        Returns:
            tuple: (transactions DataFrame, consumer_data DataFrame)
        """
        transactions = pd.read_parquet(str(DATA / "transactions" / "transactions_01.parquet"))
        consumer_data = pd.read_parquet(str(DATA / "consumer_data.parquet"))
        return transactions, consumer_data

    def __check_rep__(self):
        """Validate fit + predict on a small sample.

        Runs fit() then predict() against consumers from the first
        transaction file only, keeping instantiation fast.
        """
        transactions, consumer_data = self.load_test_case()

        # Subset to consumers present in the first transaction file
        sample_ids = transactions["masked_consumer_id"].unique()
        sample_consumers = consumer_data[
            consumer_data["masked_consumer_id"].isin(sample_ids)
        ]

        # Write the sample to a temporary parquet so fit/predict get file paths
        import tempfile, shutil
        tmpdir = tempfile.mkdtemp()
        try:
            sample_consumer_path = Path(tmpdir) / "consumer_data.parquet"
            sample_txn_dir = Path(tmpdir) / "transactions"
            sample_txn_dir.mkdir()

            sample_consumers.to_parquet(str(sample_consumer_path), index=False)
            # Copy only the first transaction file
            shutil.copy(
                str(DATA / "transactions" / "transactions_01.parquet"),
                str(sample_txn_dir / "transactions_01.parquet"),
            )

            # Fit
            try:
                self.fit(
                    str(sample_consumer_path),
                    str(sample_txn_dir),
                )
            except Exception as e:
                raise ValueError(f"fit function does not work: {e}")

            # Predict
            try:
                predicted = self.predict(
                    str(sample_consumer_path),
                    str(sample_txn_dir),
                )
            except Exception as e:
                raise ValueError(f"predict function does not work: {e}")

            # Checks
            accepted_types = (np.ndarray, list, tuple, pd.Series)
            assert isinstance(predicted, accepted_types), \
                "output should be scorable by roc_auc_score (np.ndarray, list, tuple, or pd.Series)"
            assert len(predicted) == len(sample_consumers), \
                f"each row in consumer_data needs a prediction, expected {len(sample_consumers)}, got {len(predicted)}"
            predicted_arr = np.asarray(predicted)
            assert np.all(predicted_arr >= 0) and np.all(predicted_arr <= 1), \
                "predictions should be floats between 0 and 1"

            # Score (validates that compute_score works on output)
            try:
                compute_score(sample_consumers, predicted)
            except Exception as e:
                raise ValueError(f"prediction could not be scored: {e}")
        finally:
            shutil.rmtree(tmpdir)

    @abstractmethod
    def fit(self, consumer_file: str, transactions_dir: str):
        """Train the model on the given data.

        Args:
            consumer_file: Path to consumer_data.parquet.
            transactions_dir: Path to directory containing transaction parquet files.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, consumer_file: str, transactions_dir: str) -> np.ndarray:
        """Predict financial distress probability for each consumer.

        Args:
            consumer_file: Path to consumer_data.parquet.
            transactions_dir: Path to directory containing transaction parquet files.

        Returns:
            Array of predicted probabilities (floats between 0 and 1),
            one per row in consumer_data.
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, consumer_file: str, transactions_dir: str):
        """Preprocess consumer and transaction data for the model.

        Args:
            consumer_file: Path to consumer_data.parquet.
            transactions_dir: Path to directory containing transaction parquet files.

        Returns:
            Processed features suitable for the model.
        """
        raise NotImplementedError()


def compute_score(df_consumer: pd.DataFrame, y_pred) -> float:
    """Compute mean ROC-AUC across loan types.

    Args:
        df_consumer: Consumer DataFrame with 'masked_consumer_id' and 'FPF_TARGET' columns.
        y_pred: Predicted probabilities, one per row in df_consumer.

    Returns:
        Mean AUC across loan types (grouped by first 3 chars of consumer ID).
    """
    df = df_consumer.copy()
    df['group_id'] = df['masked_consumer_id'].str[:3]
    df['y_pred'] = y_pred
    return np.mean([
        roc_auc_score(group['FPF_TARGET'], group['y_pred'])
        for _, group in df.groupby('group_id')
    ])


def score_model(model: CashflowModel, data_dir: str) -> float:
    """Score a model against holdout data.

    Args:
        model: An instantiated CashflowModel subclass.
        data_dir: Path to a data directory containing
            consumer_data.parquet and a transactions/ folder.

    Returns:
        Mean ROC-AUC across loan types.
    """
    data_dir = Path(data_dir)
    consumer_file = str(data_dir / "consumer_data.parquet")
    transactions_dir = str(data_dir / "transactions")

    consumers = pd.read_parquet(consumer_file)
    model.fit(consumer_file, transactions_dir)
    predictions = model.predict(consumer_file, transactions_dir)
    return compute_score(consumers, predictions)
