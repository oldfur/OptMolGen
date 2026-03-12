import torch
import torch.nn as nn

class LoRALinear(nn.Module):
    def __init__(self, original_linear, rank=4, alpha=0.001):
        super().__init__()
        self.original_linear = original_linear
        self.rank = rank
        # alpha 决定了 LoRA 的强度，如果发现模型改变不明显，可以调大 alpha
        self.scaling = alpha / rank
        
        in_features = original_linear.in_features
        out_features = original_linear.out_features
        
        # LoRA 权重：低秩分解 A 和 B
        # A 使用小高斯分布初始化，B 初始化为全 0
        dev = self.original_linear.weight.device
        self.lora_A = nn.Parameter(torch.randn(in_features, rank, device=dev) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=dev))
        
        # 冻结原始权重，确保训练时只更新 LoRA 参数
        self.original_linear.weight.requires_grad = False
        if self.original_linear.bias is not None:
            self.original_linear.bias.requires_grad = False

    def forward(self, x):
        # 原始冻结输出 + (x @ A @ B) * scaling
        original_out = self.original_linear(x)
        a = self.lora_A.to(x.device)
        b = self.lora_B.to(x.device)
        lora_out = (x @ a @ b) * self.scaling
        return original_out + lora_out

def inject_lora_to_last_layers(model, rank=4, alpha=4):
    """
    针对 EGNN Dynamics 的最后两个 EquivariantBlock 注入 LoRA
    同时覆盖 node_mlp (特征) 和 coord_mlp (坐标几何)
    """
    n_layers = model.n_layers
    # 针对最后两层进行“微手术”，保留前几层的全局物理规律
    target_layers = [n_layers - 1, n_layers - 2]
    
    for i in target_layers:
        block = getattr(model, f"e_block_{i}")
        
        # 1. 针对特征更新层 (node_mlp) 注入 LoRA
        # 决定了原子特征如何被 virtual_token 影响
        for j in range(block.n_layers):
            gcl = getattr(block, f"gcl_{j}")
            if isinstance(gcl.node_mlp[0], nn.Linear):
                gcl.node_mlp[0] = LoRALinear(gcl.node_mlp[0], rank=rank, alpha=alpha)
        
        # 2. 针对坐标更新层 (coord_mlp) 注入 LoRA —— 核心修改
        # 允许模型修正 virtual_token 导致的原子挤压或环变形问题
        equiv_update = block.gcl_equiv
        if isinstance(equiv_update.coord_mlp[0], nn.Linear):
            equiv_update.coord_mlp[0] = LoRALinear(equiv_update.coord_mlp[0], rank=rank, alpha=alpha)
            
    return model

def inject_lora_to_egnn(model, rank=4, alpha=4):
    """
    修正后的 EGNN LoRA 注入逻辑：
    1. 覆盖所有层的 node_mlp（特征更新），以匹配前端 FiLM 的全局调制。
    2. 完全冻结 coord_mlp（坐标更新），严格保持预训练的物理与几何等变性法则。
    """
    n_layers = model.n_layers
    
    # 【修改 1】不再只针对最后两层，而是遍历网络中的每一层 EquivariantBlock
    for i in range(n_layers):
        block = getattr(model, f"e_block_{i}")
        
        # 针对特征更新层 (node_mlp) 注入 LoRA
        # 让网络在每一层都有能力去适应 virtual_token 造成的特征空间偏移
        for j in range(block.n_layers): 
            gcl = getattr(block, f"gcl_{j}")
            if isinstance(gcl.node_mlp[0], nn.Linear):
                gcl.node_mlp[0] = LoRALinear(gcl.node_mlp[0], rank=rank, alpha=alpha)
                
            # 💡 进阶提示：如果 gcl.node_mlp 包含多个 Linear（比如 [Linear, SiLU, Linear]），
            # 且显存允许，也可以考虑遍历 node_mlp 并替换所有 nn.Linear。
            # 但替换第一层（输入层）通常是性价比最高且最稳妥的起点。

        # 【修改 2】删除了 coord_mlp 的 LoRA 注入代码
        # 核心逻辑：不去干预 equiv_update.coord_mlp。
        # 让节点特征 (h) 发生改变，但让模型依然使用预训练时学到的“受力规则”去更新坐标 (x)。
        
    return model