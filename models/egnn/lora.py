import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=1):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA 权重：低秩分解 A 和 B
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        
        # 冻结原始权重
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # 原始输出 + (x @ A @ B) * scaling
        original_out = self.original_linear(x)
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return original_out + lora_out

def inject_lora_to_last_layers(model, rank=4):
    """
    专门针对 EGNN Dynamics 的最后两个 EquivariantBlock 注入 LoRA
    """
    n_layers = model.n_layers
    target_layers = [n_layers - 1, n_layers - 2]
    
    for i in target_layers:
        block = getattr(model, f"e_block_{i}")
        # 针对每个 Block 内部的 GCL 层中的 node_mlp 进行微调
        for gcl_name in [f"gcl_{j}" for j in range(block.n_layers)]:
            gcl = getattr(block, gcl_name)
            # 包装 node_mlp 的第一个 Linear 层（通常是 hidden 变换层）
            if isinstance(gcl.node_mlp[0], nn.Linear):
                gcl.node_mlp[0] = LoRALinear(gcl.node_mlp[0], rank=rank)
    return model