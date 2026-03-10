import os
import google.generativeai as genai
from openai import OpenAI
from evaluators.base import BaseEvaluator

class GeminiEvaluator(BaseEvaluator):
    def __init__(self, model_name="gemini-1.5-flash", api_key=None):
        super().__init__(model_name)
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError("GOOGLE_API_KEY must be set for GeminiEvaluator")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model_name)
        self.history = []

    def reset(self):
        self.history = []

    def process_trial(self, image, prompt):
        self.history.append(image)
        # Gemini handles a list of PIL images + text natively
        response = self.model.generate_content(self.history + [prompt])
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

    def process_trial(self, image, prompt):
        # Implementation for GPT-4o would involve encoding images to base64
        # and sending the sequence in the messages array.
        self.history.append(image)
        # (Simplified for brevity, assuming existing helper for base64 encoding exists)
        return 1 # Placeholder for real API call logic

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
