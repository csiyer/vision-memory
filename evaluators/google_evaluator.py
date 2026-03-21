import os
from io import BytesIO
from typing import List, Dict, Any
from PIL import Image
from google import genai
from google.genai import types
from .base import BaseEvaluator


class GoogleEvaluator(BaseEvaluator):
    """Google Gemini vision evaluator."""

    def __init__(self, model_id: str = "gemini-2.0-flash"):
        super().__init__(model_id)
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _encode_image(self, image: Image.Image) -> types.Part:
        """Encode PIL Image for Gemini API."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        # Convert OpenAI-style messages to Gemini format
        contents = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if isinstance(msg["content"], str):
                parts = [msg["content"]]
            else:
                parts = []
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "text":
                        parts.append(item["text"])
                    else:
                        # Already encoded Part or image dict
                        parts.append(item)
            contents.append({"role": role, "parts": parts})

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents
        )
        return resp.text
