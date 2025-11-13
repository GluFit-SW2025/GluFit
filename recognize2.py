from ultralytics import YOLO
from pathlib import Path
import sys

MODEL_PATH = "runs/train/food_recognition/weights/best.pt"
CONFIDENCE = 0.25

# 음식 인식 및 이름 출력 (다중 객체 인식 오류로 인한 한정적 지원)
# 오류 원인 : 데이터셋 전처리시 모든 이미지의 중앙값 80%로 라벨링 영역을 잡았기 때문, 문제를 해결하려면 제대로 라벨링 영역을 처리해야 하지만, 
# 이미지 한개마다 라벨링 처리를 하기에는 너무 길다, roboflow등에서 라벨링 완료된 데이터셋을 검색해 보았지만, 마땅한 데이터셋이 없어서 AI허브 데이터셋 그대로 사용
# 따라서, 다중 객체 인식이 완벽하게 지원되지 않음, 단일 객체 인식은 높은 인식률로 정상 작동, 다만 다중객체 인식또한 일부 환경(양념, 후라이드 동시) 에서 어느정도 정상 작동하기에
# 모델을 CNN등으로 변경하지 않고 YOLO모델을 유지함
class FoodRecognizer:
    def __init__(self, model_path=MODEL_PATH):
        if not Path(model_path).exists():
            print(f"❌ 모델 파일을 찾을 수 없습니다: {model_path}")
            raise FileNotFoundError
        print(f"모델 로딩 중: {model_path}")
        self.model = YOLO(model_path)
        print("✅ 모델 로딩 완료!\n")
    
    # 이미지에서 음식을 인식하고 결과를 반환
    def recognize(self, image_path, conf=CONFIDENCE, save=True):
        if not Path(image_path).exists():
            print(f"❌ 이미지 파일을 찾을 수 없습니다: {image_path}")
            return [], None
        
        print(f"🔍 이미지 분석 중: {Path(image_path).name}")
        
        # YOLO 예측 실행 (save=True로 결과 이미지 자동 저장)
        results = self.model.predict(
            source=str(image_path),
            conf=conf,
            save=save,
            show=False,
            verbose=False,
            line_width=3,  # 박스 선 두께
            show_labels=True,  # 라벨 표시
            show_conf=True  # 신뢰도 표시
        )
        
        detected_foods = []
        save_path = None
        
        for result in results:
            # 저장된 이미지 경로
            if hasattr(result, 'save_dir') and hasattr(result, 'path'):
                save_path = Path(result.save_dir) / Path(result.path).name
            
            # 각 인식된 객체(음식) 정보 추출
            for box in result.boxes:
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                confidence = float(box.conf[0])
                x, y, w, h = box.xywhn[0].tolist()
                
                detected_foods.append({
                    'name': class_name,
                    'confidence': confidence,
                    'bbox': {'x': x, 'y': y, 'w': w, 'h': h}
                })
        
        return detected_foods, save_path
    
    # 이미지에서 음식을 인식하고 결과를 출력 
    def recognize_and_print(self, image_path, conf=CONFIDENCE):
        print("="*60)
        print(f"분석할 이미지: {Path(image_path).name}")
        print("="*60)
        
        foods, save_path = self.recognize(image_path, conf)
        
        if not foods:
            print("\n⚠️ 음식을 인식하지 못했습니다.")
            print("이미지가 흐릿하거나 음식이 명확하지 않을 수 있습니다.\n")
            return []
        # 일부 특이 케이스만 다수의 음식 인식, 보통 한개의 음식만 인식 
        print(f"\n✅ 총 {len(foods)}개의 음식을 인식했습니다!\n")
        print("-"*60)
        
        # 인식된 음식 목록 출력
        for i, food in enumerate(foods, 1):
            print(f"  [{i}] {food['name']}")
            print(f"신뢰도: {food['confidence']*100:.1f}%")
            print(f"위치: x={food['bbox']['x']:.3f}, y={food['bbox']['y']:.3f}")
            print()
        
        print("-"*60)
        
        # 결과 이미지 저장 경로 출력
        if save_path and Path(save_path).exists():
            print(f"\n결과 이미지(영역 저장) 저장됨: {save_path}")
        
        return foods

# 사용자로부터 이미지 경로를 입력받는 함수
def get_user_input():
    print("\n" + "="*60)
    print("음식 인식 프로그램 (다중 객체 인식)")
    print("="*60)
    print("\n 분석할 이미지 파일 경로를 입력하세요.")
    print("  (예: ./test_image.jpg, /home/user/food.png)")
    print("  종료하려면 'q' 또는 'quit'를 입력하세요.\n")
    
    while True:
        image_path = input("이미지 경로 >>> ").strip()
        
        # 종료 명령어 확인
        if image_path.lower() in ['q', 'quit']:
            print("\n 프로그램을 종료합니다.\n")
            sys.exit(0)
        
        # 경로가 비어있으면 다시 입력 받기
        if not image_path:
            print("⚠️ 경로를 입력해주세요.\n")
            continue
        
        # 파일 존재 여부 확인
        if Path(image_path).exists():
            return image_path
        else:
            print(f"❌ 파일을 찾을 수 없습니다: {image_path}")
            print("   경로를 다시 확인해주세요.\n")


if __name__ == "__main__":
    try:
        # 모델 로딩
        recognizer = FoodRecognizer()
        
        # 사용자 입력 모드
        while True:
            # 이미지 경로 입력 받기
            image_path = get_user_input()
            
            # 음식 인식 및 결과 출력
            recognizer.recognize_and_print(image_path, conf=CONFIDENCE)
            
            print("\n" + "="*60 + "\n")
            
    except KeyboardInterrupt:
        print("\n사용자에 의해 종료되었습니다.\n")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}\n")
        sys.exit(1)