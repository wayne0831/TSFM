import torch
import torch.nn as nn
import math
import time
from config import *

###########################################################################################################
# 1. LoRA Layer: 초기값 안정화
###########################################################################################################

class LoRALayer(nn.Module):
    def __init__(self, original_layer, rank=8, alpha=32, dropout=0.1):
        super().__init__()
        self.original_layer = original_layer
        self.original_layer.weight.requires_grad = False
        
        in_features = original_layer.in_features
        out_features = original_layer.out_features
        
        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, rank))
        self.scaling = alpha / rank 
        self.dropout = nn.Dropout(p=dropout)

        # 초기 상태에서 원본 모델과 동일하게 출력되도록 B를 0으로 초기화
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        result = self.original_layer(x)
        # 차원 에러 방지를 위해 dropout 위치 확인
        lora_out = (self.dropout(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scaling
        return result + lora_out

###########################################################################################################
# 2. LoRA 적용 함수: 모듈 경로 검사 강화 (Tokenizer 완벽 차단)
###########################################################################################################

def apply_lora_to_tsfm(model, target_modules=['qkv_proj', 'out'], rank=4, alpha=32, dropout=0.1):
    # 전체 파라미터 freeze
    for param in model.parameters():
        param.requires_grad = False

    core_model = model.model if hasattr(model, 'model') else model

    applied_count = 0
    for name, module in core_model.named_modules():
        # [핵심] Linear 레이어이면서, 이름에 'stacked_xf'가 있고, 'tokenizer'가 절대 없어야 함
        if isinstance(module, nn.Linear):
            is_target = any(t == name.split('.')[-1] or t in name for t in target_modules)
            is_in_transformer = 'stacked_xf' in name
            is_not_tokenizer = 'tokenizer' not in name
            
            if is_target and is_in_transformer and is_not_tokenizer:
                parent_path = name.split('.')
                parent = core_model
                for part in parent_path[:-1]:
                    parent = getattr(parent, part)
                
                target_name = parent_path[-1]
                # 교체
                setattr(parent, target_name, LoRALayer(module, rank, alpha, dropout))
                applied_count += 1

    # 학습 대상 파라미터만 requires_grad 활성화
    tr_params = 0
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
            tr_params += param.numel()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ LoRA applied to {applied_count} layers.")
    print(f"📊 Trainable: {tr_params:,} / Total: {total_params:,} ({tr_params/total_params:.4%})")
    
    return model, tr_params/total_params

###########################################################################################################
# 3. Train 함수: Loss 계산 및 학습 모드 강제
###########################################################################################################

def train(model, train_loader, max_horizon, patch_size, lr, epochs):
    p_size = int(patch_size)
    core_model = model.model if hasattr(model, 'model') else model
    
    # 훈련 모드 설정 및 그래디언트 활성화
    core_model.train()
    torch.set_grad_enabled(True)

    lora_params = [p for n, p in core_model.named_parameters() if "lora_" in n and p.requires_grad]
    
    # Loss 정체 해결을 위해 가중치 감쇠(weight_decay) 조정 및 LR 최적화
    optimizer = torch.optim.AdamW(lora_params, lr=lr)
    criterion = nn.MSELoss()
    history = []

    print(f"\n🚀 학습 시작 (LR: {lr})...")

    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch_idx, (batch_x, batch_y, batch_mask) in enumerate(train_loader):
            batch_x, batch_y = batch_x.to(DEVICE), batch_y.to(DEVICE)
            optimizer.zero_grad()
            
            num_patches = batch_x.shape[1] // p_size
            x_reshaped = batch_x.view(-1, num_patches, p_size)
            x_input = x_reshaped[:, :, :p_size-1].clone().detach().requires_grad_(True)
            mask_1 = torch.zeros(x_reshaped.shape[0], num_patches, 1).to(DEVICE)
            
            # Forward
            try:
                outputs = model(x_input, mask_1)
            except:
                outputs = core_model(x_input, mask_1)

            while isinstance(outputs, (tuple, list)):
                outputs = outputs[0]

            # 그래디언트 전파 확인
            if outputs.grad_fn is None:
                raise RuntimeError("🚨 Gradient chain broken! Check LoRA injection.")

            pred_all = outputs.reshape(batch_x.shape[0], -1)
            
            # 타겟 데이터(batch_y)의 스케일에 맞춰 예측값 슬라이싱
            # TimesFM 출력의 마지막 max_horizon 부분이 batch_y와 대응함
            pred_last = pred_all[:, -max_horizon:]
            target = batch_y[:, :max_horizon]
            
            loss = criterion(pred_last, target)
            loss.backward()

            # 그래디언트 클리핑으로 수치 안정성 확보
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(train_loader)
        print(f"🔥 Epoch [{epoch+1}/{epochs}] Avg Loss: {avg_loss:.6f}")
        history.append(avg_loss)
    
    # 평가 모드 전환
    core_model.eval()
    return model, history