import torch
from ultralytics import YOLO
import DCNv4
print('DCNv4 엔진 정상 작동!')
print('CUDA 버전:', torch.version.cuda)

model = YOLO('yolov8n.pt')
