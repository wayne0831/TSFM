###########################################################################################################
# import libraries
###########################################################################################################

import torch
import torch.nn as nn
import math
import time
from config import *

###########################################################################################################
# define LoRA layer
###########################################################################################################

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1):
        super().__init__()

        self.original_layer = original_layer
        # 기존 가중치 동결
        self.original_layer.weight.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        # LoRA 파라미터 선언 및 학습 가능 상태 설정
        self.A = nn.Parameter(torch.empty(rank, in_features), requires_grad=True)
        self.B = nn.Parameter(torch.empty(out_features, rank), requires_grad=True)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        # 가중치 초기화
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        # 1. 기존 레이어 통과 (Frozen 경로)
        result = self.original_layer(x)
        
        # 2. LoRA 경로 연산 (Trainable 경로)
        # x @ A.t() 연산이 시작될 때 A가 requires_grad=True이므로 
        # 전체 결과물에 grad_fn이 생성되어 backward가 가능해집니다.
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        
        return result + lora_out

# 2. apply_lora_to_tsfm 함수
def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    # 기초 모델의 모든 파라미터 동결
    for param in model.parameters():
        param.requires_grad = False

    # 타겟 모듈을 LoRALayer로 교체
    for i, block in enumerate(model.stacked_xf):
        if 'qkv_proj' in target_modules:
            block.attn.qkv_proj = LoRALayer(block.attn.qkv_proj, rank, alpha, dropout)
        if 'out' in target_modules:
            block.attn.out = LoRALayer(block.attn.out, rank, alpha, dropout)
        if 'ff0' in target_modules:
            block.ff0 = LoRALayer(block.ff0, rank, alpha, dropout)
        if 'ff1' in target_modules:
            block.ff1 = LoRALayer(block.ff1, rank, alpha, dropout)
    
    # [핵심 추가] 새로 삽입된 LoRA 파라미터(A, B)의 학습 가능 상태를 강제로 활성화
    # 모델 전체를 얼린 후 레이어를 교체했기 때문에 명시적으로 True를 설정하는 것이 안전합니다.
    for name, param in model.named_parameters():
        if ".A" in name or ".B" in name:
            param.requires_grad = True

    print(f"Apply LoRA: Target Modules -> {target_modules}")

    total_params = sum(p.numel() for p in model.parameters())
    tr_params    = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable Parameters: {tr_params:,} | Ratio: {tr_params/total_params:.2%}")

    return model, tr_params/total_params

def remove_lora_from_tsfm(model):
    """LoRALayer를 다시 원래의 nn.Linear(original_layer)로 되돌립니다."""
    for i, block in enumerate(model.stacked_xf):
        # qkv_proj 복구
        if isinstance(block.attn.qkv_proj, LoRALayer):
            block.attn.qkv_proj = block.attn.qkv_proj.original_layer
        # out 복구
        if isinstance(block.attn.out, LoRALayer):
            block.attn.out = block.attn.out.original_layer
        # ff0 복구
        if hasattr(block, 'ff0') and isinstance(block.ff0, LoRALayer):
            block.ff0 = block.ff0.original_layer
        # ff1 복구
        if hasattr(block, 'ff1') and isinstance(block.ff1, LoRALayer):
            block.ff1 = block.ff1.original_layer
    print("✅ LoRA layers reverted to original Linear layers.")
    return model

def train(model, train_loader, max_horizon, patch_size, lr, epochs):
    p_size = int(patch_size)
    
    # 학습 대상(LoRA 파라미터만) 추출
    tr_params = [p for p in model.model.parameters() if p.requires_grad]
    
    if len(tr_params) == 0:
        raise RuntimeError("No trainable parameters found! Check if LoRA was applied correctly.")

    optimizer = torch.optim.AdamW(tr_params, lr=lr)
    criterion = nn.MSELoss()

    model.model.train() # 학습 모드 설정
    history = []

    print(f"\n🚀 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (batch_x, batch_y, batch_mask) in enumerate(train_loader):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            num_patches = batch_x.shape[1] // p_size
            x_reshaped = batch_x.view(-1, num_patches, p_size)
            
            x_63 = x_reshaped[:, :, :p_size-1]
            # x_63이 LoRA 레이어에 도달하기 전까지 연산 그래프를 유지하도록 설정
            x_63.requires_grad_(True) 
            
            mask_1 = torch.zeros(x_reshaped.shape[0], num_patches, 1).to(DEVICE)
            
            outputs = model.model(x_63, mask_1)
            
            while isinstance(outputs, (tuple, list)): 
                outputs = outputs[0]
            
            if outputs.dim() == 4:
                pred_all = outputs[:, :, :, 0].reshape(batch_x.shape[0], -1)
            else:
                pred_all = outputs.reshape(batch_x.shape[0], -1)
                
            pred_last = pred_all[:, -max_horizon:]
            
            if pred_last.shape[1] != batch_y.shape[1]:
                loss = criterion(pred_last, batch_y[:, :pred_last.shape[1]])
            else:
                loss = criterion(pred_last, batch_y)

            # 이제 loss는 grad_fn을 가지므로 backward()가 정상 작동합니다.
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        history.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_epoch_loss:.6f} | Time: {epoch_duration:.2f}s")
    
    return model, history