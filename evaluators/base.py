from abc import ABC, abstractmethod
from typing import List, Dict, Any
from pathlib import Path
from PIL import Image

# Load .env file from project root
try:
    from dotenv import load_dotenv
    env_path = Path(__file__).parent.parent / '.env'
    load_dotenv(env_path)
except ImportError:
    pass  # dotenv not installed, rely on environment variables


class BaseEvaluator(ABC):
    """Base class for VLM evaluators."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    @abstractmethod
    def _encode_image(self, image: Image.Image) -> Any:
        """Encode PIL Image to API-specific format."""
        pass

    @abstractmethod
    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        pass

    def get_name(self) -> str:
        return self.model_id
