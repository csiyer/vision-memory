import torch
from typing import List, Dict, Any
from PIL import Image
from .base import BaseEvaluator


class QwenEvaluator(BaseEvaluator):
    """Qwen3-VL local inference evaluator.

    Requires: transformers, torch, qwen_vl_utils
    GPU recommended for reasonable inference speed.
    """

    def __init__(self, model_id: str = "Qwen/Qwen3-VL-8B-Instruct"):
        super().__init__(model_id)
        self.model = None
        self.processor = None
        self._load_model()

    def _load_model(self):
        """Load Qwen3-VL model and processor."""
        from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

        print(f"Loading {self.model_id}...")

        # Use bfloat16 for better memory efficiency on GPU
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(
            self.model_id,
            dtype=torch.bfloat16,
            device_map="auto",
        )

        self.processor = AutoProcessor.from_pretrained(self.model_id)
        print(f"Model loaded on {next(self.model.parameters()).device}")

    def _encode_image(self, image: Image.Image) -> Dict[str, Any]:
        """Return image in Qwen-compatible format.

        Qwen expects images as PIL Images in the message content.
        """
        # Resize large images to save memory
        max_size = 512
        if max(image.size) > max_size:
            image = image.copy()
            image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)

        return {"type": "image", "image": image}

    def _call_api(self, messages: List[Dict]) -> str:
        """Run local inference and return response text."""
        # Convert messages to Qwen format
        qwen_messages = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if isinstance(content, str):
                qwen_messages.append({"role": role, "content": content})
            else:
                # Build content list with text and images
                qwen_content = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get("type") == "text":
                            qwen_content.append({"type": "text", "text": item["text"]})
                        elif item.get("type") == "image":
                            qwen_content.append({"type": "image", "image": item["image"]})
                        elif item.get("type") == "image_url":
                            # Handle base64 encoded images (convert back to PIL)
                            import base64
                            from io import BytesIO
                            url = item["image_url"]["url"]
                            if url.startswith("data:"):
                                b64_data = url.split(",")[1]
                                img_bytes = base64.b64decode(b64_data)
                                img = Image.open(BytesIO(img_bytes))
                                qwen_content.append({"type": "image", "image": img})
                    else:
                        qwen_content.append(item)

                qwen_messages.append({"role": role, "content": qwen_content})

        # Process with Qwen processor
        text_prompt = self.processor.apply_chat_template(
            qwen_messages,
            tokenize=False,
            add_generation_prompt=True
        )

        # Collect all images from messages
        images = []
        for msg in qwen_messages:
            if isinstance(msg["content"], list):
                for item in msg["content"]:
                    if isinstance(item, dict) and item.get("type") == "image":
                        images.append(item["image"])

        # Process inputs
        if images:
            inputs = self.processor(
                text=[text_prompt],
                images=images,
                padding=True,
                return_tensors="pt"
            )
        else:
            inputs = self.processor(
                text=[text_prompt],
                padding=True,
                return_tensors="pt"
            )

        inputs = inputs.to(self.model.device)

        # Generate response
        with torch.no_grad():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=100,
                do_sample=False
            )

        # Decode only the new tokens
        generated_ids = output_ids[:, inputs.input_ids.shape[1]:]
        response = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return response.strip()

    def get_name(self) -> str:
        """Return short name for results."""
        model_upper = self.model_id.upper()
        if "8B" in model_upper:
            return "qwen3-vl-8b"
        elif "4B" in model_upper:
            return "qwen3-vl-4b"
        return "qwen3-vl"
