###########################################################################################################
# import libraries
###########################################################################################################

from xml.parsers.expat import model

import torch
import torch.nn as nn
import math
import time
from config import *

###########################################################################################################
# import libraries
###########################################################################################################

# define LoRA layer
class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=16, dropout=0.1): # 당시 droput 오타 상태
        super().__init__()

        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.A = nn.Parameter(torch.empty(rank, in_features), requires_grad=True)
        self.B = nn.Parameter(torch.empty(out_features, rank), requires_grad=True)
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(p=dropout)

        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def forward(self, x):
        result = self.original_layer(x)
        lora_out = (self.dropout(x) @ self.A.t() @ self.B.t()) * self.scaling
        return result + lora_out

# 2. apply_lora_to_tsfm 함수
def apply_lora_to_tsfm(model, target_modules=['qkv_proj'], rank=4, alpha=16, dropout=0.1):
    # freeze all parameters of the foundation model
    for param in model.parameters():
        param.requires_grad = False

    for i, block in enumerate(model.stacked_xf):
        if 'qkv_proj' in target_modules:
            block.attn.qkv_proj = LoRALayer(block.attn.qkv_proj, rank, alpha)
        if 'out' in target_modules:
            block.attn.out = LoRALayer(block.attn.out, rank, alpha, dropout)
        if 'ff0' in target_modules:
            block.ff0 = LoRALayer(block.ff0, rank, alpha, dropout)
        if 'ff1' in target_modules:
            block.ff1 = LoRALayer(block.ff1, rank, alpha, dropout)
    
    print(f"Apply LoRA: Target Modules -> {target_modules}")

    total_params = sum(p.numel() for p in model.parameters())
    tr_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total Parameters: {total_params:,} | Trainable Parameters: {tr_params:,} | Ratio: {tr_params/total_params:.2%}")

    return model


def train(model, train_loader, max_horizon, patch_size, lr, epochs):
    # 정수형 보장 (view 에러 방지)
    p_size = int(patch_size)
    
    # 학습 대상(LoRA 파라미터만) 추출
    tr_params = [p for p in model.model.parameters() if p.requires_grad]
    
    if len(tr_params) == 0:
        raise RuntimeError("No trainable parameters found! Check if LoRA was applied correctly.")

    optimizer = torch.optim.AdamW(tr_params, lr=lr)
    criterion = nn.MSELoss()

    model.model.train()
    history = []

    print(f"\n🚀 Starting training for {epochs} epochs...")

    for epoch in range(epochs):
        epoch_start_time = time.time()
        epoch_loss = 0.0
        
        for batch_idx, (batch_x, batch_y, batch_mask) in enumerate(train_loader):
            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            
            # 패치 단위 리쉐이핑
            num_patches = batch_x.shape[1] // p_size
            x_reshaped = batch_x.view(-1, num_patches, p_size)
            
            # TimesFM 입력 (64 -> 63)
            x_63 = x_reshaped[:, :, :p_size-1]
            mask_1 = torch.zeros(x_reshaped.shape[0], num_patches, 1).to(DEVICE)
            
            # Forward
            outputs = model.model(x_63, mask_1)
            
            while isinstance(outputs, (tuple, list)): 
                outputs = outputs[0]
            
            # 예측값 정제 및 Horizon 슬라이싱
            if outputs.dim() == 4:
                pred_all = outputs[:, :, :, 0].reshape(batch_x.shape[0], -1)
            else:
                pred_all = outputs.reshape(batch_x.shape[0], -1)
                
            pred_last = pred_all[:, -max_horizon:]
            
            # Loss 계산 및 역전파
            if pred_last.shape[1] != batch_y.shape[1]:
                loss = criterion(pred_last, batch_y[:, :pred_last.shape[1]])
            else:
                loss = criterion(pred_last, batch_y)

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        avg_epoch_loss = epoch_loss / len(train_loader)
        epoch_duration = time.time() - epoch_start_time
        history.append(avg_epoch_loss)
        
        print(f"Epoch [{epoch+1}/{epochs}] Loss: {avg_epoch_loss:.6f} | Time: {epoch_duration:.2f}s")
    
    return model, history