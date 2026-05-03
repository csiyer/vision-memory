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

    def check_image_capacity(self, n_images: int) -> bool:
        """Probe whether the model can handle n_images in a single request.

        Sends n_images 1x1 white pixel images with a minimal prompt and catches
        any 400/413/context-too-large errors from the provider.
        Returns True if the probe succeeds, False if the model rejects it.
        """
        pixel = Image.new("RGB", (1, 1), (255, 255, 255))
        encoded = [self._encode_image(pixel) for _ in range(n_images)]
        content = [{"type": "text", "text": "Reply with the number 1."}] + encoded
        messages = [{"role": "user", "content": content}]
        try:
            self._call_api(messages)
            return True
        except Exception as e:
            msg = str(e).lower()
            if any(tok in msg for tok in ("400", "413", "too large", "too many", "image", "limit", "exceed", "invalid")):
                return False
            raise
