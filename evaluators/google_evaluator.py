import os
import time
from io import BytesIO
from typing import Any, Dict, List
from PIL import Image
from google import genai
from google.genai import types
from google.genai.errors import ClientError, ServerError
from .base import BaseEvaluator


class GoogleEvaluator(BaseEvaluator):
    """Google Gemini vision evaluator."""

    def __init__(self, model_id: str = "gemini-3.1-flash-image-preview"):
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

        time.sleep(35)  # free tier: 10 RPM shared across ~5 concurrent jobs
        for attempt in range(16):
            try:
                resp = self.client.models.generate_content(
                    model=self.model_id,
                    contents=contents
                )
                return resp.text
            except ClientError as e:
                if e.code == 429 and attempt < 15:
                    wait = min(30 * (2 ** attempt), 600)  # cap at 10 min
                    print(f"\n  Gemini 429 rate limit, waiting {wait}s (attempt {attempt+1}/16)...")
                    time.sleep(wait)
                else:
                    raise
            except ServerError as e:
                if attempt < 15:
                    wait = min(30 * (2 ** attempt), 600)
                    print(f"\n  Gemini 5xx ({e.code}), waiting {wait}s (attempt {attempt+1}/16)...")
                    time.sleep(wait)
                else:
                    raise

    def check_image_capacity(self, n_images: int) -> bool:
        """Override to skip the rate-limit sleep during the probe."""
        from PIL import Image as PILImage
        pixel = PILImage.new("RGB", (1, 1), (255, 255, 255))
        encoded = [self._encode_image(pixel) for _ in range(n_images)]
        parts = [types.Part.from_text(text="Reply with the number 1.")] + encoded
        contents = [types.Content(role="user", parts=parts)]
        try:
            self.client.models.generate_content(model=self.model_id, contents=contents)
            return True
        except ClientError as e:
            if e.code in (400, 413):
                return False
            raise
