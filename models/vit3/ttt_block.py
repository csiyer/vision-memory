import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_
import torch.nn.functional as F

class TTT(nn.Module):
    """ Robust Test-Time Training block using actual autograd for memory persistence.
    """
    def __init__(self, dim, num_heads, qkv_bias=True, **kwargs):
        super().__init__()
        head_dim = dim // num_heads
        self.dim = dim
        self.num_heads = num_heads

        self.qkv = nn.Linear(dim, dim * 3 + head_dim * 3, bias=qkv_bias)
        self.w1 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w2 = nn.Parameter(torch.zeros(1, self.num_heads, head_dim, head_dim))
        self.w3 = nn.Parameter(torch.zeros(head_dim, 1, 3, 3))
        trunc_normal_(self.w1, std=.02)
        trunc_normal_(self.w2, std=.02)
        trunc_normal_(self.w3, std=.02)
        self.proj = nn.Linear(dim + head_dim, dim)
        self.scale = (9)**-0.5

    def forward(self, x, h, w, state=None, learning_rate=0.001, return_grad_norm=False):
        b, n, c = x.shape
        d = c // self.num_heads

        q1, k1, v1, q2, k2, v2 = torch.split(self.qkv(x), [c, c, c, d, d, d], dim=-1)
        q1 = q1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        k1 = k1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        v1 = v1.reshape(b, n, self.num_heads, d).transpose(1, 2)
        
        q2 = q2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        k2 = k2.reshape(b, h, w, d).permute(0, 3, 1, 2)
        v2 = v2.reshape(b, h, w, d).permute(0, 3, 1, 2)

        # 1. LOAD PREVIOUS WEIGHTS (MEM)
        # We clone to avoid mutating the core parameters during the probe
        mem_w1 = state['w1'].clone().detach() if state else self.w1.clone().detach()
        mem_w2 = state['w2'].clone().detach() if state else self.w2.clone().detach()
        mem_w3 = state['w3'].clone().detach() if state else self.w3.clone().detach()

        # 2. PERFORM TTT UPDATE (Memory Imprinting)
        total_ttt_loss = 0.0
        grad_norm = 0.0
        # If learning_rate > 0, we perform one actual step on the current (k, v)
        if learning_rate > 0 or return_grad_norm:
            with torch.enable_grad():
                mem_w1.requires_grad_(True)
                mem_w2.requires_grad_(True)
                mem_w3.requires_grad_(True)
                
                # Forward on TTT task
                f_k1 = (k1 @ mem_w1) * F.silu(k1 @ mem_w2)
                loss1 = F.mse_loss(f_k1, v1)
                
                f_k2 = F.conv2d(k2.reshape(1, b * d, h, w), mem_w3.repeat(b,1,1,1), padding=1, groups=b * d)
                loss2 = F.mse_loss(f_k2, v2.reshape(1, b * d, h, w))
                
                total_ttt_loss = (loss1 + loss2) / 2.0
                
                # Backprop (Single Step)
                grads = torch.autograd.grad(total_ttt_loss, [mem_w1, mem_w2, mem_w3], allow_unused=True)
                
                if return_grad_norm:
                    grad_norm = sum(torch.norm(g).item() for g in grads if g is not None)

                if learning_rate > 0:
                    with torch.no_grad():
                        if grads[0] is not None: 
                            g = grads[0] / (torch.norm(grads[0]) + 1.0)
                            mem_w1 = mem_w1 - learning_rate * g
                        if grads[1] is not None: 
                            g = grads[1] / (torch.norm(grads[1]) + 1.0)
                            mem_w2 = mem_w2 - learning_rate * g
                        if grads[2] is not None: 
                            g = grads[2] / (torch.norm(grads[2]) + 1.0)
                            mem_w3 = mem_w3 - learning_rate * g
                
                # Detach to clear the graph since we are in a sequence
                mem_w1 = mem_w1.detach()
                mem_w2 = mem_w2.detach()
                mem_w3 = mem_w3.detach()
        else:
            # Just compute loss for probing
            with torch.no_grad():
                f_k1 = (k1 @ mem_w1) * F.silu(k1 @ mem_w2)
                loss1 = F.mse_loss(f_k1, v1)
                f_k2 = F.conv2d(k2.reshape(1, b * d, h, w), mem_w3.repeat(b,1,1,1), padding=1, groups=b * d)
                loss2 = F.mse_loss(f_k2, v2.reshape(1, b * d, h, w))
                total_ttt_loss = (loss1 + loss2) / 2.0

        # 3. APPLY UPDATED WEIGHTS TO QUERY (Retrieval)
        x1 = (q1 @ mem_w1) * F.silu(q1 @ mem_w2)
        x1 = x1.transpose(1, 2).reshape(b, n, c)
        
        x2 = F.conv2d(q2.reshape(1, b * d, h, w), mem_w3.repeat(b,1,1,1), padding=1, groups=b * d)
        x2 = x2.reshape(b, d, n).transpose(1, 2)

        x = torch.cat([x1, x2], dim=-1)
        x = self.proj(x)

        new_state = {'w1': mem_w1, 'w2': mem_w2, 'w3': mem_w3}
        metrics = {'ttt_loss': total_ttt_loss, 'grad_norm': grad_norm}
        
        return x, new_state, metrics
