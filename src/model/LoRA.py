###########################################################################################################
# import libraries
###########################################################################################################

import torch
import torch.nn as nn
import math

###########################################################################################################
# import libraries
###########################################################################################################

# define LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, droput=0.1):
        super().__init__()
        
        # freeze original layer parameters
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        # set diemnsions for low-rank matrices
        in_features  = original_layer.in_features
        out_features = original_layer.out_features
        
        # set low-rank matrices A and B
        self.A  = nn.Parameter(torch.empty(rank, in_features))
        self.B  = nn.Parameter(torch.empty(out_features, rank))
        self.scaling = alpha / rank
        
        # set dropout for LoRA
        self.dropout = nn.Dropout(p=droput)

        # initialize low-rank matrices
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # result of original layer
        result = self.original_layer(x)

        # result of LoRA (only on last dimension)
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        
        return result + lora_out

# apply LoRA to target layers in TimesFM
def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    for i, block in enumerate(model.stacked_xf):
        if 'qkv_proj' in target_modules:
            block.attn.qkv_proj = LoRALayer(block.attn.qkv_proj, rank, alpha)
        if 'out' in target_modules:
            block.attn.out = LoRALayer(block.attn.out, rank, alpha, dropout)
        if 'ff0' in target_modules:
            block.ff0 = LoRALayer(block.ff0, rank, alpha, dropout)
        if 'ff1' in target_modules:
            block.ff1 = LoRALayer(block.ff1, rank, alpha, dropout)
        # end if
    
    # print target modules for verification
    print(f"Apply LoRA: Target Modules -> {target_modules}")

    # print the number of total and trainable parameters for verification
    total_params = sum(p.numel() for p in model.parameters())
    tr_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable Parameters: {tr_params:,} | Ratio: {tr_params/total_params:.2%}")

    return model