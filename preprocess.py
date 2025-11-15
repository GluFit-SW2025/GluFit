import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm


class DatasetPreprocessor:
    def __init__(self, source_dir, target_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
        
        #source_dir: 전처리할 데이터 저장된 경로
        #target_dir: 전처리된 데이터를 저장할 경로
        #train_ratio: 학습 데이터 비율 (0.8)
        #val_ratio: 검증 데이터 비율 (0.1)
        #test_ratio: 테스트 데이터 비율 (0.1)
        #seed: 42

        self.source_dir = Path(source_dir)
        self.target_dir = Path(target_dir)
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        
        random.seed(seed)
        
        # 목표 디렉토리 생성
        self.train_dir = self.target_dir / 'train'
        self.val_dir = self.target_dir / 'validation'
        self.test_dir = self.target_dir / 'test'
        
        print(f'원본 데이터: {self.source_dir}')
        print(f'저장 경로: {self.target_dir}')
        print(f'분할 비율 - Train: {train_ratio}, Val: {val_ratio}, Test: {test_ratio}')
    
    def get_all_food_classes(self):
        food_classes = []
        
        # AI Hub 데이터 구조: 대분류/소분류/이미지.jpg
        for big_category in self.source_dir.iterdir():
            if big_category.is_dir():
                for small_category in big_category.iterdir():
                    if small_category.is_dir():
                        food_name = small_category.name
                        food_classes.append({
                            'name': food_name,
                            'path': small_category,
                            'big_category': big_category.name
                        })
        
        print(f'\n✓ 총 {len(food_classes)}개 음식 클래스 발견')
        return food_classes
    
    # 유효 이미지 파일 필터링 함수
    def get_valid_images(self, folder_path):
        valid_extensions = {'.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG'}
        images = []
        
        for img_path in folder_path.iterdir():
            if img_path.suffix in valid_extensions:
                # 파일 크기 체크 (0바이트 파일 제외)
                if img_path.stat().st_size > 0:
                    images.append(img_path)
        
        return images
    
    # 이미지를 train/val/test로 분할하고 복사
    def split_and_copy_images(self, food_class):
        food_name = food_class['name']
        source_path = food_class['path']
        
        # 모든 이미지 파일 가져오기
        images = self.get_valid_images(source_path)
        
        if len(images) == 0:
            print(f'⚠️  {food_name}: 이미지 없음')
            return 0, 0, 0
        
        # 랜덤 셔플
        random.shuffle(images)
        
        # 분할 인덱스 계산
        total = len(images)
        train_end = int(total * self.train_ratio)
        val_end = train_end + int(total * self.val_ratio)
        
        train_images = images[:train_end]
        val_images = images[train_end:val_end]
        test_images = images[val_end:]
        
        # 각 폴더에 복사
        splits = {
            'train': (self.train_dir, train_images),
            'validation': (self.val_dir, val_images),
            'test': (self.test_dir, test_images)
        }
        
        for split_name, (split_dir, split_images) in splits.items():
            # 음식별 폴더 생성
            food_folder = split_dir / food_name
            food_folder.mkdir(parents=True, exist_ok=True)
            
            # 이미지 복사
            for img_path in split_images:
                dest_path = food_folder / img_path.name
                shutil.copy2(img_path, dest_path)
        
        return len(train_images), len(val_images), len(test_images)
    
    # 전체 전처리 실행
    def process(self):
        print('\n' + '='*70)
        print('데이터셋 전처리 시작')
        print('='*70)
        
        # 기존 데이터 폴더 확인
        if self.target_dir.exists():
            response = input(f'\n⚠️  {self.target_dir} 폴더가 이미 존재합니다. 삭제하고 진행할까요? (y/n): ')
            if response.lower() == 'y':
                shutil.rmtree(self.target_dir)
                print('기존 폴더 삭제 완료')
            else:
                print('취소되었습니다.')
                return
        
        # 목표 디렉토리 생성
        self.train_dir.mkdir(parents=True, exist_ok=True)
        self.val_dir.mkdir(parents=True, exist_ok=True)
        self.test_dir.mkdir(parents=True, exist_ok=True)
        
        # 음식 클래스 목록 가져오기
        food_classes = self.get_all_food_classes()
        
        # 각 음식별로 처리
        print('\n이미지 분할 및 복사 중')
        total_train = 0
        total_val = 0
        total_test = 0
        
        for food_class in tqdm(food_classes, desc='Processing'):
            train_cnt, val_cnt, test_cnt = self.split_and_copy_images(food_class)
            total_train += train_cnt
            total_val += val_cnt
            total_test += test_cnt
        
        # 결과 출력
        print('\n' + '='*70)
        print('전처리 완료!')
        print('='*70)
        print(f'Train 이미지:      {total_train:,}개')
        print(f'Validation 이미지: {total_val:,}개')
        print(f'Test 이미지:       {total_test:,}개')
        print(f'총 이미지:         {total_train + total_val + total_test:,}개')
        print(f'\n저장 위치: {self.target_dir.absolute()}')
        print('='*70)
        
        # 디렉토리 구조 확인
        print('\n생성된 디렉토리 구조:')
        print(f'{self.target_dir}/')
        print(f'├── train/ ({len(list(self.train_dir.iterdir()))} 클래스)')
        print(f'├── validation/ ({len(list(self.val_dir.iterdir()))} 클래스)')
        print(f'└── test/ ({len(list(self.test_dir.iterdir()))} 클래스)')


def main():
    
    # 경로 입력
    print('\n1. 원본 데이터 경로를 입력하세요.')
    source_dir = input('\n원본 데이터 경로: ').strip()
    
    print('\n2. 전처리된 데이터를 저장할 경로를 입력하세요.')
    target_dir = input('\n저장 경로: ').strip()
    
    # 경로 확인
    if not os.path.exists(source_dir):
        print(f'\n❌ 오류: {source_dir} 경로가 존재하지 않습니다.')
        return
    
    # 전처리 실행
    preprocessor = DatasetPreprocessor(
        source_dir=source_dir,
        target_dir=target_dir,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42
    )
    
    preprocessor.process()
    
    print('\n✅ 전처리가 완료되었습니다!')
    print(f'이제 train.py에서 base_dir = "{target_dir}" 로 설정하고 학습을 시작하세요.')


if __name__ == '__main__':
    main()