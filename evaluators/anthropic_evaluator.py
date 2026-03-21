import os
import base64
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from anthropic import Anthropic
from .base import BaseEvaluator


class AnthropicEvaluator(BaseEvaluator):
    """Anthropic Claude vision evaluator."""

    def __init__(self, model_id: str = "claude-sonnet-4-20250514"):
        super().__init__(model_id)
        self.client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    def _encode_image(self, image: Image.Image) -> Dict[str, Any]:
        """Encode PIL Image to base64 for Anthropic API."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": b64
            }
        }

    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        resp = self.client.messages.create(
            model=self.model_id,
            messages=messages,
            max_tokens=10
        )
        return resp.content[0].text
