'''
📌 모델 평가 스크립트 (test_model.py)
'''


from ultralytics import YOLO
import yaml
import os
import torch
from ultralytics import YOLO

# ================= [여기만 수정하세요] =================
# 1. 학습된 모델 경로 (best.pt)
MODEL_PATH = '/home/haggi/DCNv4/runs/detect/yolo11n_DCNv4/train2/weights/best.pt'


# 2. 평가할 테스트 이미지 폴더 (test/images)
TEST_IMAGES_DIR = '/home/haggi/fire_detection_datasets/origin_datasets/merged_origin_datasets/test/images'
temp_yaml_path = '/home/haggi/fire_detection_datasets/origin_datasets/merged_origin_datasets/test/test.yaml'

# 3. 클래스 정보 (학습할 때 썼던 data.yaml에 있는 내용 그대로)
CLASS_NAMES = {
    0: 'fire',
    1: 'smoke'
}
# ======================================================

def evaluate_model():
    print(f"🚀 테스트 데이터셋 평가 시작... (경로: {TEST_IMAGES_DIR})")
    
    # 2. 모델 로드
    model = YOLO(MODEL_PATH)

    # 3. 검증(Validation) 모드로 실행하지만, 데이터는 Test셋임
    # conf=0.001은 mAP 계산용 표준값입니다.
    metrics = model.val(task="detection",data=temp_yaml_path, split='val', conf=0.001, verbose=True,save_json=True,project = "validation_res")

    # 4. 핵심 지표 출력
    print("\n" + "="*30)
    print("📊 [최종 성적표] 📊")
    print(f"🔥 mAP 50    (감지 능력): {metrics.box.map50:.4f}")
    print(f"🎯 mAP 50-95 (정밀 능력): {metrics.box.map:.4f}")
    print(f"🔫 Precision (정밀도)  : {metrics.box.mp:.4f}")
    print(f"👀 Recall    (재현율)  : {metrics.box.mr:.4f}")
    print("="*30)

    print(f"\n✅ 상세 결과(오답노트, 그래프)는 여기 저장됨:")
    print(f"👉 {metrics.save_dir}")
    

if __name__ == '__main__':
    evaluate_model()

