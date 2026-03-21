import os
import base64
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI
from .base import BaseEvaluator


class OpenAIEvaluator(BaseEvaluator):
    """OpenAI GPT vision evaluator."""

    def __init__(self, model_id: str = "gpt-4o"):
        super().__init__(model_id)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _encode_image(self, image: Image.Image, detail: str = "low") -> Dict[str, Any]:
        """Encode PIL Image to base64 for OpenAI API."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        b64 = base64.b64encode(buf.getvalue()).decode()
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{b64}",
                "detail": detail
            }
        }

    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        resp = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=10
        )
        return resp.choices[0].message.content
