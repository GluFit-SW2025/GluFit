import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import torch.nn as nn
import os

# 학습된 모델 로드 (num_classes 자동 감지, 모델 학습 후 저장시 부족한 부분 정의 보완 버전)
def load_model(checkpoint_path, device='auto'):
    # Device 설정
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)

    # Checkpoint 로드
    checkpoint = torch.load(checkpoint_path, map_location=device)

    print("\n=== 모델 파일 정보 ===")
    print("keys:", checkpoint.keys())

    # config 가져오기
    config = checkpoint.get("config", {})
    model_type = config.get("model_type", "mobilenet")
    img_size = config.get("img_size", 224)

    # === num_classes 자동 감지 ===
    state_dict = checkpoint["model_state_dict"]

    # mobilenet head 기준
    if "classifier.1.weight" in state_dict:
        num_classes = state_dict["classifier.1.weight"].shape[0]
        print(f"✓ num_classes 자동 감지됨: {num_classes}")
    else:
        # fallback
        num_classes = config.get("num_classes", 50)
        print(f"⚠ classifier.weight 없음 → config num_classes 사용: {num_classes}")

    # class_names 처리
    class_names = checkpoint.get("class_names", [f"class_{i}" for i in range(num_classes)])
    print(f"✓ class_names count = {len(class_names)}")

    # 모델 생성
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(in_features, num_classes)
    )

    # 가중치 로드
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    # 전처리
    transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    print(f"✓ 모델 로드 완료 → {model_type}, classes={num_classes}, img={img_size}")
    return model, transform, class_names, device

# 이미지 예측 함수
def predict_image(image_path, model, transform, class_names, device, top_k=5):

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(img_tensor)
        probs = F.softmax(outputs, dim=1)[0]

    top_probs, top_idx = torch.topk(probs, top_k)

    return [
        (class_names[idx], float(prob))
        for idx, prob in zip(top_idx, top_probs)
    ]


# 메인 루프 made in GPT
if __name__ == '__main__':
    print('\n' + '=' * 70)
    print('  한식 이미지 분류 시스템')
    print('=' * 70)

    print('\n모델 파일 경로를 입력하세요.')
    print('예시: checkpoints/mobilenet/best_model.pth')
    checkpoint_path = input('모델 경로: ').strip()

    # 모델 로드
    try:
        model, transform, class_names, device = load_model(checkpoint_path)
    except Exception as e:
        print("\n❌ 모델 로드 실패:")
        print(e)
        exit(1)

    # 예측
    while True:
        print('\n' + '-' * 70)
        print('예측할 이미지 경로를 입력하세요. (종료: q or exit)')
        image_path = input('이미지 경로: ').strip()

        if image_path.lower() in ['q','exit']:
            print("\n프로그램 종료.")
            break

        if not os.path.exists(image_path):
            print(f"\n❌ 파일 없음: {image_path}")
            continue

        try:
            print(f"\n예측 중: {image_path}")
            results = predict_image(image_path, model, transform, class_names, device)

            print("\n🎯 Top-5 예측 결과")
            print('-' * 70)
            for i, (name, prob) in enumerate(results, 1):
                bar = '█' * int(prob * 40)
                print(f'{i}. {name:20s}  {prob:6.2%}  {bar}')

            print(f"\n✓ 최종 예측: {results[0][0]}")
            print(f"✓ 신뢰도: {results[0][1]:.2%}")

        except Exception as e:
            print("\n❌ 예측 중 오류 발생:")
            print(e)
