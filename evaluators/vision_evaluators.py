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
        try:
            from mambavision import create_model
            self.model = create_model(model_name, pretrained=True).to(device)
            self.model.eval()
        except ImportError:
            print("Warning: 'mambavision' library not found. Run !pip install mambavision")
            self.model = None
            
        self.prev_state = None

    def process_trial(self, image, prompt=None):
        if self.model is None:
            return 0.5 # Fallback if library missing
            
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # For MambaVision, we treat the cumulative state change as the match signal.
            # In zero-shot, we can look at the output features or the internal state diff.
            features = self.model.forward_features(img_tensor)
            
            # Simple surprisal metric: how much did this image change our representation?
            if self.prev_state is None:
                self.prev_state = features
                return 0.0
            
            # Cosine similarity between current and previous representation
            # Lower similarity = higher 'Newness' / 'Surprise'
            # We return (1 - similarity) as a score where higher = more 'repeat-like' 
            # (Wait, actually higher similarity = more repeat-like)
            score = F.cosine_similarity(features, self.prev_state).item()
            self.prev_state = features
            return float(score)

class TitansNeuralMemory(torch.nn.Module):
    """A small MLP that acts as the long-term memory for Titans."""
    def __init__(self, dim=768):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Linear(dim, dim)
        )
    def forward(self, x):
        return self.net(x)

class VisionTitansEvaluator(RecurrentVisionEvaluator):
    def __init__(self, model_name="vit_base_patch16_224", device="cuda"):
        super().__init__("vision_titans", device)
        import timm
        # 1. Official Pre-trained 'Eyes'
        self.backbone = timm.create_model(model_name, pretrained=True).to(device)
        self.backbone.eval()
        
        # 2. Titans Nonlinear Neural Memory (MLP)
        self.dim = 768
        self.memory_net = TitansNeuralMemory(self.dim).to(device)
        self.optimizer = torch.optim.SGD(self.memory_net.parameters(), lr=0.1)
        
    def reset(self):
        # Re-initialize weights to 'blank slate' for a new sequence
        for layer in self.memory_net.net:
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def process_trial(self, image, prompt=None):
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            # A. Extract Features
            features = self.backbone.forward_features(img_tensor)
            if hasattr(self.backbone, 'global_pool') and self.backbone.global_pool:
                x = self.backbone.forward_head(features, pre_logits=True)
            else:
                x = features[:, 0]
            x = F.normalize(x, p=2, dim=-1) # [1, dim]

        # B. Probe Memory (Recognition)
        # We check how well the MLP can reconstruct the current signal BEFORE learning it.
        # High reconstruction accuracy = High recognition.
        self.memory_net.eval()
        prediction = self.memory_net(x)
        # Cosine similarity as match score (0 to 1)
        match_score = F.cosine_similarity(x, prediction).item()
        
        # C. Titans 'Learning to Memorize' Update
        # We perform 1 step of gradient descent to 'store' the current image in the MLPs weights.
        self.memory_net.train()
        self.optimizer.zero_grad()
        output = self.memory_net(x)
        loss = F.mse_loss(output, x)
        loss.backward()
        self.optimizer.step()
        
        return float(match_score)
