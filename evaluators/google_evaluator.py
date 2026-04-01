import os
from io import BytesIO
from typing import Any, Dict, List
from PIL import Image
from google import genai
from google.genai import types
from .base import BaseEvaluator


class GoogleEvaluator(BaseEvaluator):
    """Google Gemini vision evaluator."""

    def __init__(self, model_id: str = "gemini-2.5-flash"):
        super().__init__(model_id)
        self.client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY"))

    def _encode_image(self, image: Image.Image) -> types.Part:
        """Encode PIL Image for Gemini API."""
        buf = BytesIO()
        image.save(buf, format="JPEG")
        return types.Part.from_bytes(data=buf.getvalue(), mime_type="image/jpeg")

    def _to_part(self, item: Any) -> types.Part:
        """Convert OpenAI-style content items to google.genai Part instances."""
        if isinstance(item, types.Part):
            return item
        if isinstance(item, str):
            return types.Part.from_text(text=item)
        if isinstance(item, dict) and item.get("type") == "text":
            return types.Part.from_text(text=item["text"])
        return types.Part(item)

    def _call_api(self, messages: List[Dict]) -> str:
        """Make API call and return response text."""
        contents: list[types.Content] = []
        for msg in messages:
            role = "user" if msg["role"] == "user" else "model"
            if isinstance(msg["content"], str):
                parts = [self._to_part(msg["content"])]
            else:
                parts = [self._to_part(item) for item in msg["content"]]
            contents.append(types.Content(role=role, parts=parts))

        resp = self.client.models.generate_content(
            model=self.model_id,
            contents=contents
        )
        return resp.text
