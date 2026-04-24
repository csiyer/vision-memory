import os
import time
import base64
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from openai import OpenAI, RateLimitError
from .base import BaseEvaluator


class OpenAIEvaluator(BaseEvaluator):
    """OpenAI GPT vision evaluator."""

    def __init__(self, model_id: str = "gpt-4o"):
        super().__init__(model_id)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))

    def _encode_image(self, image: Image.Image, detail: str = "low", max_size: int = 512) -> Dict[str, Any]:
        """Encode PIL Image to base64 for OpenAI API.

        Args:
            image: PIL Image to encode
            detail: 'low' (85 tokens) or 'high' (more tokens, better quality)
            max_size: Resize image to fit within this dimension (reduces tokens)
        """
        # Resize to reduce token usage
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        buf = BytesIO()
        image.save(buf, format="JPEG", quality=85)
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
        for attempt in range(8):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model_id,
                    messages=messages,
                    max_tokens=20
                )
                return resp.choices[0].message.content
            except RateLimitError:
                if attempt < 7:
                    wait = 30 * (2 ** attempt)  # 30, 60, 120, 240 ... seconds
                    print(f"\n  GPT 429 rate limit, waiting {wait}s (attempt {attempt+1}/8)...")
                    time.sleep(wait)
                else:
                    raise
