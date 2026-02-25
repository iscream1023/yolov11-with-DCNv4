from ultralytics import YOLO
import torch
from torch.profiler import profile, record_function, ProfilerActivity

# 1. 모델 로드
torch.set_default_tensor_type('torch.cuda.FloatTensor')
yolo = YOLO('/home/haggi/DCNv4/yolo_v11n_DCNv4.yaml') # 혹은 사용자님의 커스텀 모델
model = yolo.model.cuda().eval() # nn.Module만 추출 및 평가 모드
model.to('cuda')
torch.set_default_tensor_type('torch.FloatTensor') # 복구

# 2. 더미 입력 (YOLO 기본 해상도 640)
inputs = torch.randn(1, 3, 640, 640).cuda()

for name, param in model.named_parameters():
    if not param.is_cuda:
        print(f"⚠️ 경고: {name} 레이어가 CPU에 있습니다! 강제 이동합니다.")
        param.data = param.data.to('cuda')

for name, buf in model.named_buffers():
    if not buf.is_cuda:
        print(f"🚨 버퍼 발견! [{name}]가 CPU에 있습니다. 강제 이동합니다.")
        buf.data = buf.data.cuda()

# 3. Warm-up (GPU 예열 - 이거 안 하면 첫 로딩 시간이 포함되어 수치 망함)
for _ in range(10):
    _ = model(inputs)

# 4. 진짜 측정
with profile(
    activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], # CPU를 넣어야 "inference" 라벨이 보입니다.
    record_shapes=True,
    with_stack=False # 속도 측정을 위해 스택은 끄는 게 좋습니다.
) as prof:
    with record_function("inference"):
        # 연산 시작
        output = model(inputs)
        # GPU 연산이 끝날 때까지 CPU가 기다리게 함 (정확한 시간 측정의 핵심)
        torch.cuda.synchronize()

# 5. 레이어별 시간 출력
# DCNv4나 C3k2라는 이름이 들어간 연산자를 찾으세요
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=30))