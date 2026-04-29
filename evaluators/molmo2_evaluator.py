import torch
from typing import List, Dict, Any
from PIL import Image
from .base import BaseEvaluator


class Molmo2Evaluator(BaseEvaluator):
    """Molmo2-8B local inference evaluator.

    Requires: transformers, torch
    GPU recommended for reasonable inference speed.
    """

    def __init__(self, model_id: str = "allenai/Molmo2-8B"):
        super().__init__(model_id)
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Molmo2 model and processor."""
        import os
        from transformers import AutoModelForImageTextToText, AutoProcessor

        print(f"Loading {self.model_id}...")

        offline = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
        kwargs = dict(trust_remote_code=True, dtype="auto", device_map="auto")
        if offline:
            kwargs["local_files_only"] = True

        try:
            self.processor = AutoProcessor.from_pretrained(self.model_id, **kwargs)
            self.model = AutoModelForImageTextToText.from_pretrained(self.model_id, **kwargs)
        except (OSError, Exception) as e:
            if offline:
                raise RuntimeError(
                    f"Model '{self.model_id}' not found in HuggingFace cache "
                    f"({os.environ.get('HF_HOME', '~/.cache/huggingface')}) and "
                    f"TRANSFORMERS_OFFLINE=1 prevents downloading. "
                    f"Run this on a login node first:\n"
                    f"  python3 -c \"from transformers import AutoProcessor, AutoModelForImageTextToText; "
                    f"AutoProcessor.from_pretrained('{self.model_id}', trust_remote_code=True); "
                    f"AutoModelForImageTextToText.from_pretrained('{self.model_id}', trust_remote_code=True, dtype='auto')\""
                ) from e
            raise

        print(f"Model loaded on {next(self.model.parameters()).device}")

    def _encode_image(self, image: Image.Image) -> Dict[str, Any]:
        """Return image in Molmo2-compatible format.

        Molmo2 expects PIL Images passed via apply_chat_template.
        """
        max_size = 512
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return {"type": "image", "image": image}

    def _call_api(self, messages: List[Dict]) -> str:
        """Run local Molmo2 inference and return response text."""
        molmo_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                molmo_messages.append({"role": role, "content": [{"type": "text", "text": content}]})
            else:
                molmo_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            molmo_content.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image":
                            molmo_content.append({"type": "image", "image": item["image"]})
                        elif item.get("type") == "image_url":
                            import base64
                            from io import BytesIO
                            url = item["image_url"]["url"]
                            if url.startswith("data:"):
                                b64_data = url.split(",")[1]
                                img_bytes = base64.b64decode(b64_data)
                                img = Image.open(BytesIO(img_bytes))
                                molmo_content.append({"type": "image", "image": img})
                    else:
                        molmo_content.append(item)
                molmo_messages.append({"role": role, "content": molmo_content})

        inputs = self.processor.apply_chat_template(
            molmo_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True,
        )

        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False,
            )

        # Decode only new tokens
        generated_tokens = generated_ids[0, inputs["input_ids"].shape[1]:]
        response = self.processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response.strip()

    def get_name(self) -> str:
        """Return short name for results."""
        return "molmo2-8b"
