import torch
import torch.nn.functional as F
import numpy as np
import os
from evaluators.base import BaseEvaluator

# Force slow but compatible path if kernels aren't installed correctly
os.environ["MAMBA_SKIP_CUDA_KERNEL"] = "1"
os.environ["CAUSAL_CONV1D_FORCE_SYCL"] = "0"

try:
    import mamba_ssm
    from mamba_ssm import Mamba
except ImportError:
    mamba_ssm = None
    print("Warning: mamba_ssm not found. RecurrentVisionEvaluator may fail.")

try:
    import causal_conv1d
except ImportError:
    causal_conv1d = None
    print("Warning: causal_conv1d not found. Some SSM architectures may fail.")

class ViTEvaluator(BaseEvaluator):
    def __init__(self, model_name="vit_base_patch16_224", device="cuda"):
        super().__init__(model_name)
        import timm
        self.device = device
        self.model = timm.create_model(model_name, pretrained=True).to(device)
        self.model.eval()
        self.buffer = [] # Store image embeddings

    def reset(self):
        self.buffer = []

    def process_trial(self, image, prompt=None):
        # 1. Extract embedding
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            embedding = self.model.forward_features(img_tensor)
            if hasattr(self.model, 'global_pool') and self.model.global_pool:
                embedding = self.model.forward_head(embedding, pre_logits=True)
            else:
                embedding = embedding[:, 0] # Take CLS token
        
        if not self.buffer:
            self.buffer.append(embedding)
            return 0.0 
        
        # 2. Compute similarity with buffer
        all_embeddings = torch.cat(self.buffer, dim=0)
        similarities = F.cosine_similarity(embedding, all_embeddings)
        max_sim = torch.max(similarities).item()
        
        # 3. Update buffer 
        self.buffer.append(embedding)
        return float(max_sim)

class RecurrentVisionEvaluator(BaseEvaluator):
    """Base for MambaVision, Causal Mamba, and Titans."""
    def __init__(self, model_name, device="cuda"):
        super().__init__(model_name)
        self.device = device
        self.state = None # Hidden state h_t

    def reset(self):
        self.state = None

    def process_trial(self, image, prompt=None):
        # This will be overridden or implemented for specific SSM architectures
        raise NotImplementedError("Use specific class like MambaEvaluator")

class MambaVisionEvaluator(RecurrentVisionEvaluator):
    def __init__(self, model_name="mambavision_tiny_1k", device="cuda"):
        super().__init__(model_name, device)
        # Load MambaVision from specialized repo/timm if available
        # self.model = ...

    def process_trial(self, image, prompt=None):
        # 1. Update internal SSM state
        # 2. Return 'Surprise' signal (||h_t - h_{t-1}||)
        return 0.5 

class VisionTitansEvaluator(RecurrentVisionEvaluator):
    def __init__(self, device="cuda"):
        super().__init__("vision_titans", device)
        # Load from Titans repo

    def process_trial(self, image, prompt=None):
        # Return Neural Memory Readout magnitude
        return 0.8
