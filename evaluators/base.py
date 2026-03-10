from abc import ABC, abstractmethod

class BaseEvaluator(ABC):
    """
    Base class for all model evaluators in the continuous recognition task.
    """
    def __init__(self, model_name):
        self.model_name = model_name

    @abstractmethod
    def reset(self):
        """Reset internal model state/history between sequences."""
        pass

    @abstractmethod
    def process_trial(self, image, prompt):
        """
        Processes a single trial and returns a match score.
        For VLMs: This could be 0 or 1.
        For Vision Models: This could be a float (similarity/surprise).
        """
        pass
        
    def get_name(self):
        return self.model_name
