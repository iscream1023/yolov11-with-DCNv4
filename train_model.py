from ultralytics import YOLO
import torch
from torchvision import models,datasets,transforms

torch.cuda.empty_cache()
if torch.cuda.is_available():
    print(f"@Success@ cuda is available @Success@")
else:
    print(f"@Warning@ cuda is unavailable @Warning@")
    exit(1)

#model = YOLO('yolo11s.pt')
#model.load('/home/haggi/make_model/runs/detect/yolo11s/weights/best.pt')
model = YOLO('/home/haggi/DCNv4/yolo_v11n_DCNv4.yaml') 
model.load('yolo11n.pt')
model.to('cuda')

origin_datasets_root = r"/home/haggi/fire_detection_datasets/origin_datasets/merged_origin_datasets/data.yaml"
small_datasets_root = r"/home/haggi/fire_detection_datasets/origin_datasets/mini_datasets/data.yaml"
seg_datasets_root = r"/home/haggi/fire_detection_datasets/AI-HUB segmentation/yolo_final/data.yaml"

def main():

    model.train(
        data=origin_datasets_root,
        lr0=5e-4,
        warmup_epochs = 3.0,
        project="yolo11n_DCNv4",
        epochs=50,
        device=0,
        patience=10,
        workers=8,
        imgsz=640,
    )

if __name__ == "__main__":  
    main()