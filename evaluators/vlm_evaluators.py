import os
import base64
from io import BytesIO
from google import genai
from openai import OpenAI
from evaluators.base import BaseEvaluator

class GeminiEvaluator(BaseEvaluator):
    def __init__(self, model_name="gemini-1.5-flash", api_key=None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set for GeminiEvaluator")
        
        self.client = genai.Client(api_key=self.api_key)
        self.history = []

    def reset(self):
        self.history = []

    def process_trial(self, image, prompt):
        self.history.append(image)
        # Gemini handles a list of PIL images + text natively in the new SDK
        # We send the full sequence of images seen so far + the prompt
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=self.history + [prompt]
        )
        text = response.text.lower()
        return 1 if "yes" in text else 0

class OpenAIEvaluator(BaseEvaluator):
    def __init__(self, model_name="gpt-4o", api_key=None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY must be set for OpenAIEvaluator")
        
        self.client = OpenAI(api_key=self.api_key)
        self.history = []

    def reset(self):
        self.history = []

    def _pil_to_base64(self, img):
        buffered = BytesIO()
        img.save(buffered, format="JPEG")
        return base64.b64encode(buffered.getvalue()).decode('utf-8')

    def process_trial(self, image, prompt):
        self.history.append(image)
        
        # Prepare the message content with all images in history
        content = []
        for img in self.history:
            b64_img = self._pil_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
            })
        
        content.append({"type": "text", "text": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": content}],
            max_tokens=10
        )
        
        text = response.choices[0].message.content.lower()
        return 1 if "yes" in text else 0

class LocalVLMEvaluator(BaseEvaluator):
    """Evaluator for models like InternVL2 or Qwen2-VL using transformers."""
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name)
        from transformers import AutoModel, AutoTokenizer
        self.device = device
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        self.history = []

    def reset(self):
        self.history = []

    def process_trial(self, image, prompt):
        self.history.append(image)
        # Real local inference logic using self.model and self.tokenizer
        return 1
