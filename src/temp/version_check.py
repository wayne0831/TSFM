import torch

print(f"CUDA 사용 가능 여부: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"현재 GPU 이름: {torch.cuda.get_device_name(0)}")
    print(f"현재 사용 중인 장치 번호: {torch.cuda.current_device()}")