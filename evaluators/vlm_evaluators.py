import os
import base64
from io import BytesIO
from google import genai
from openai import OpenAI
from evaluators.base import BaseEvaluator

class GeminiEvaluator(BaseEvaluator):
    def __init__(self, model_id="gemini-1.5-pro"):
        super().__init__(model_id)
        api_key = os.environ.get("GOOGLE_API_KEY")
        if not api_key:
            print("Warning: GOOGLE_API_KEY not found in environment.")
        self.client = genai.Client(api_key=api_key)
        self.model_id = model_id
        self.chat = None # Persistent chat session
        
    def reset(self):
        # Clears the session memory for a new 100-image run
        self.chat = self.client.chats.create(model=self.model_id)

    def process_trial(self, image, prompt):
        # We only send the SINGLE current image. 
        # The history is maintained by the 'chat' object.
        response = self.chat.send_message([prompt, image])
        txt = response.text.lower()
        return 1.0 if "yes" in txt else 0.0

class OpenAIEvaluator(BaseEvaluator):
    def __init__(self, model_id="gpt-4o-mini"):
        super().__init__(model_id)
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        self.model_id = model_id
        self.history = []
        
    def reset(self):
        self.history = []

    def process_trial(self, image, prompt):
        # Convert PIL to Base64 for the API
        buffered = BytesIO()
        image.save(buffered, format="JPEG")
        base64_image = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # Append ONLY the current image/prompt to the history
        self.history.append({
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
            ]
        })

        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=self.history,
            max_tokens=10,
        )
        ans = response.choices[0].message.content.lower()
        
        # Add the model's answer to history to maintain conversational context
        self.history.append({"role": "assistant", "content": ans})
        
        return 1.0 if "yes" in ans else 0.0

class Qwen2VLEvaluator(BaseEvaluator):
    def __init__(self, model_id="Qwen/Qwen2-VL-7B-Instruct"):
        super().__init__("qwen2-vl-7b")
        from transformers import Qwen2VLForConditionalGeneration, AutoProcessor, BitsAndBytesConfig
        from torch import bfloat16
        
        # Correct way to load in 4-bit using BitsAndBytesConfig
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            model_id, 
            torch_dtype=bfloat16, 
            device_map="auto", 
            quantization_config=quant_config
        )
        self.processor = AutoProcessor.from_pretrained(model_id)
        self.history = []

    def reset(self):
        self.history = []

    def process_trial(self, image, prompt):
        from qwen_vl_utils import process_vision_info
        self.history.append({
            "role": "user",
            "content": [{"type": "image", "image": image}, {"type": "text", "text": prompt}]
        })
        
        text = self.processor.apply_chat_template(self.history, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(self.history)
        inputs = self.processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
        inputs = inputs.to(self.model.device)

        generated_ids = self.model.generate(**inputs, max_new_tokens=10)
        generated_ids_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        ans = self.processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0].lower()
        
        self.history.append({"role": "assistant", "content": ans})
        return 1.0 if "yes" in ans else 0.0

class InternVLEvaluator(BaseEvaluator):
    def __init__(self, model_id="OpenGVLab/InternVL2-8B"):
        super().__init__("internvl2-8b")
        from transformers import AutoModel, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.model = AutoModel.from_pretrained(
            model_id, 
            torch_dtype=torch.bfloat16, 
            low_cpu_mem_usage=True, 
            trust_remote_code=True, 
            device_map="auto", 
            quantization_config=quant_config
        ).eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        self.history = None 
        
    def reset(self):
        self.history = None

    def process_trial(self, image, prompt):
        import torch
        from torchvision import transforms
        
        # InternVL2 image processing
        transform = transforms.Compose([
            transforms.Resize((448, 448), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        pixel_values = transform(image).unsqueeze(0).to(torch.bfloat16).to(self.model.device)
        
        # InternVL2 Chat interface
        response, history = self.model.chat(
            self.tokenizer, 
            pixel_values, 
            prompt, 
            generation_config={"max_new_tokens": 10}, 
            history=self.history,
            return_history=True
        )
        self.history = history 
        ans = response.lower()
        return 1.0 if "yes" in ans else 0.0
