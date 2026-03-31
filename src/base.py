"""Abstract base class for scorable competition models.

Each competition defines a concrete subclass that implements:
    - load_test_case()   — returns (raw_input, true_output) for self-testing
    - predict()          — runs the model on raw inputs
    - process_inputs()   — preprocesses raw data into model features
    - compute_score()    — scores predictions against ground truth

Students inherit the competition-specific subclass (e.g. EnhancerModel)
and implement predict() and process_inputs(). The __check_rep__ mechanism
validates their implementation at instantiation time.
"""
from abc import ABC, abstractmethod


class ScorableModelTemplate(ABC):
    """Abstract class for scorable models.

    Subclasses should implement predict() and process_inputs().
    A test case is run when a subclass is instantiated to validate
    the implementation.
    """

    def __init__(self):
        """Validate the subclass implementation on instantiation.

        Subclasses that define __init__ should call super().__init__()
        at the end, after setting up any attributes needed by __check_rep__.
        """
        self.__check_rep__()

    @abstractmethod
    def load_test_case(self):
        """Load a test case for validation.

        Returns:
            tuple: (raw_input, true_output) where formats are competition-specific.
        """
        raise NotImplementedError()

    @abstractmethod
    def predict(self, *args, **kwargs):
        """Run the model on raw inputs.

        Signature varies by competition.
        """
        raise NotImplementedError()

    @abstractmethod
    def process_inputs(self, *args, **kwargs):
        """Preprocess raw data into model features.

        Signature varies by competition.
        """
        raise NotImplementedError()

    def __check_rep__(self):
        """Validate that predict() works and output is scorable.

        Subclasses should override this to add competition-specific checks.
        """
        pass
