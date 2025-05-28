import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import requests
import zipfile
import io
import glob
import cv2
from tqdm import tqdm
import gdown
import shutil

# MPS 디바이스 확인 및 설정
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"사용 중인 디바이스: {device}")

# WFLW 데이터셋 다운로드 함수
def download_wflw_dataset():
    """WFLW(Wider Facial Landmarks in the Wild) 데이터셋 다운로드 함수"""
    base_dir = 'data/wflw'
    os.makedirs(base_dir, exist_ok=True)
    
    # 이미 다운로드 되었는지 확인
    if os.path.exists(os.path.join(base_dir, 'WFLW_images')) and \
       os.path.exists(os.path.join(base_dir, 'WFLW_annotations')):
        print("WFLW 데이터셋이 이미 존재합니다. 다운로드를 건너뜁니다.")
        return True
    
    print("WFLW 데이터셋 다운로드 중...")
    
    # WFLW 데이터셋 다운로드 URL (Google Drive)
    # 이미지 다운로드
    image_url = "https://drive.google.com/uc?id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC"
    # 어노테이션 다운로드
    anno_url = "https://drive.google.com/uc?id=1-tW0L0CDPhMycZL0IW3ZjOBQfw-D9yU3"
    
    try:
        # 이미지 다운로드
        print("WFLW 이미지 다운로드 중...")
        image_zip = os.path.join(base_dir, "wflw_images.zip")
        gdown.download(image_url, image_zip, quiet=False)
        
        # 어노테이션 다운로드
        print("WFLW 어노테이션 다운로드 중...")
        anno_zip = os.path.join(base_dir, "wflw_annotations.zip")
        gdown.download(anno_url, anno_zip, quiet=False)
        
        # 압축 해제
        print("이미지 압축 해제 중...")
        with zipfile.ZipFile(image_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        print("어노테이션 압축 해제 중...")
        with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
            zip_ref.extractall(base_dir)
        
        # 임시 파일 삭제
        os.remove(image_zip)
        os.remove(anno_zip)
        
        print("WFLW 데이터셋 다운로드 및 설정 완료")
        return True
    except Exception as e:
        print(f"WFLW 데이터셋 다운로드 중 오류 발생: {e}")
        print("대체 방법으로 다시 시도합니다...")
        
        # 대체 방법: 직접 다운로드
        try:
            import urllib.request
            
            # 이미지 다운로드
            print("WFLW 이미지 직접 다운로드 중...")
            image_zip = os.path.join(base_dir, "wflw_images.zip")
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?export=download&id=1hzBd48JIdWTJSsATBEB_eFVvPL1bx6UC",
                image_zip
            )
            
            # 어노테이션 다운로드
            print("WFLW 어노테이션 직접 다운로드 중...")
            anno_zip = os.path.join(base_dir, "wflw_annotations.zip")
            urllib.request.urlretrieve(
                "https://drive.google.com/uc?export=download&id=1-tW0L0CDPhMycZL0IW3ZjOBQfw-D9yU3",
                anno_zip
            )
            
            # 압축 해제
            print("이미지 압축 해제 중...")
            with zipfile.ZipFile(image_zip, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            
            print("어노테이션 압축 해제 중...")
            with zipfile.ZipFile(anno_zip, 'r') as zip_ref:
                zip_ref.extractall(base_dir)
            
            # 임시 파일 삭제
            os.remove(image_zip)
            os.remove(anno_zip)
            
            print("WFLW 데이터셋 다운로드 및 설정 완료")
            return True
        except Exception as e2:
            print(f"대체 다운로드 방법도 실패: {e2}")
            print("테스트용 더미 데이터셋을 생성합니다...")
            return create_dummy_dataset()

# 테스트용 더미 데이터셋 생성
def create_dummy_dataset():
    """테스트용 간단한 랜드마크 데이터셋 생성"""
    base_dir = 'data/wflw'
    os.makedirs(os.path.join(base_dir, 'WFLW_images'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'WFLW_annotations'), exist_ok=True)
    
    num_landmarks = 98  # WFLW는 98개 랜드마크 사용
    num_samples = 100   # 100개 샘플 생성
    
    # 훈련, 테스트 폴더 생성
    train_dir = os.path.join(base_dir, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'train')
    test_dir = os.path.join(base_dir, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'test')
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 훈련, 테스트 파일 생성
    train_file = os.path.join(train_dir, 'list.txt')
    test_file = os.path.join(test_dir, 'list.txt')
    
    # 이미지 폴더
    image_dir = os.path.join(base_dir, 'WFLW_images')
    
    with open(train_file, 'w') as f_train, open(test_file, 'w') as f_test:
        for i in range(num_samples):
            # 더미 이미지 생성 (200x200 크기)
            img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            
            # 얼굴 영역
            cx, cy = 100, 100
            face_w, face_h = 120, 150
            
            # 얼굴 타원 그리기
            cv2.ellipse(img, (cx, cy), (face_w//2, face_h//2), 0, 0, 360, (220, 220, 220), -1)
            cv2.ellipse(img, (cx, cy), (face_w//2, face_h//2), 0, 0, 360, (0, 0, 0), 2)
            
            # 눈, 코, 입 등 간단한 특징 추가
            cv2.circle(img, (cx-25, cy-20), 10, (200, 200, 200), -1)  # 왼쪽 눈
            cv2.circle(img, (cx+25, cy-20), 10, (200, 200, 200), -1)  # 오른쪽 눈
            cv2.circle(img, (cx-25, cy-20), 10, (0, 0, 0), 1)  # 왼쪽 눈 윤곽
            cv2.circle(img, (cx+25, cy-20), 10, (0, 0, 0), 1)  # 오른쪽 눈 윤곽
            cv2.circle(img, (cx, cy+10), 8, (150, 150, 150), -1)  # 코
            cv2.ellipse(img, (cx, cy+40), (20, 10), 0, 0, 180, (100, 100, 100), -1)  # 입
            
            # 이미지 파일명
            img_filename = f"dummy_{i:04d}.jpg"
            img_path = os.path.join(image_dir, img_filename)
            cv2.imwrite(img_path, img)
            
            # 랜드마크 생성 (98개 포인트를 얼굴 주변에 분포)
            landmarks = []
            
            # 얼굴 윤곽선 (33개 포인트)
            for j in range(33):
                angle = j * (360 / 33) * np.pi / 180
                r = face_w // 2
                x = cx + int(r * np.cos(angle))
                y = cy + int((face_h/face_w) * r * np.sin(angle))
                landmarks.append((x, y))
            
            # 왼쪽 눈썹 (9개 포인트)
            for j in range(9):
                x = cx - 35 + j * 5
                y = cy - 40 + int(3 * np.sin(j * np.pi / 8))
                landmarks.append((x, y))
            
            # 오른쪽 눈썹 (9개 포인트)
            for j in range(9):
                x = cx + 5 + j * 5
                y = cy - 40 + int(3 * np.sin(j * np.pi / 8))
                landmarks.append((x, y))
            
            # 왼쪽 눈 (8개 포인트)
            for j in range(8):
                angle = j * (360 / 8) * np.pi / 180
                r = 10
                x = cx - 25 + int(r * np.cos(angle))
                y = cy - 20 + int(r * np.sin(angle))
                landmarks.append((x, y))
            
            # 오른쪽 눈 (8개 포인트)
            for j in range(8):
                angle = j * (360 / 8) * np.pi / 180
                r = 10
                x = cx + 25 + int(r * np.cos(angle))
                y = cy - 20 + int(r * np.sin(angle))
                landmarks.append((x, y))
            
            # 코 (12개 포인트)
            for j in range(12):
                if j < 4:  # 코 상단
                    x = cx - 6 + j * 4
                    y = cy - 5 + j
                elif j < 8:  # 코 중간
                    x = cx - 8 + (j-4) * 5
                    y = cy + 5
                else:  # 코 하단
                    x = cx - 10 + (j-8) * 7
                    y = cy + 10
                landmarks.append((x, y))
            
            # 입 (12개 외곽선 포인트)
            for j in range(12):
                angle = j * (360 / 12) * np.pi / 180
                r_x = 20
                r_y = 10
                x = cx + int(r_x * np.cos(angle))
                y = cy + 40 + int(r_y * np.sin(angle))
                landmarks.append((x, y))
            
            # 입 (7개 내부 포인트)
            for j in range(7):
                if j < 4:  # 윗입술
                    x = cx - 12 + j * 8
                    y = cy + 35
                else:  # 아랫입술
                    x = cx - 12 + (j-4) * 12
                    y = cy + 45
                landmarks.append((x, y))
            
            # 랜드마크 좌표가 98개인지 확인
            if len(landmarks) != num_landmarks:
                # 부족하면 채우기
                while len(landmarks) < num_landmarks:
                    landmarks.append((cx, cy))
                # 많으면 자르기
                landmarks = landmarks[:num_landmarks]
            
            # 파일에 쓰기 (WFLW 포맷 - 랜드마크 좌표, 경계 상자, 속성)
            landmark_str = ' '.join([f"{x} {y}" for x, y in landmarks])
            # 경계 상자 (x, y, w, h)
            bbox = f"{cx-face_w//2} {cy-face_h//2} {face_w} {face_h}"
            # 더미 속성 (표정, 포즈, 조명, 화장 등 총 6개)
            attrs = "0 0 0 0 0 0"
            
            # 전체 라인 구성
            line = f"{landmark_str} {bbox} {attrs} {img_filename}\n"
            
            # 80%는 훈련, 20%는 테스트 데이터로
            if i < int(num_samples * 0.8):
                f_train.write(line)
            else:
                f_test.write(line)
    
    print(f"테스트용 더미 WFLW 데이터셋 생성 완료: {num_samples}개 샘플")
    return True


# WFLW 데이터셋 클래스
class WFLWDataset(Dataset):
    def __init__(self, root_dir, is_train=True, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.is_train = is_train
        
        # 이미지 디렉토리
        self.image_dir = os.path.join(root_dir, 'WFLW_images')
        
        # 어노테이션 파일 경로
        if is_train:
            anno_file = os.path.join(root_dir, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'train', 'list.txt')
        else:
            anno_file = os.path.join(root_dir, 'WFLW_annotations', 'list_98pt_rect_attr_train_test', 'test', 'list.txt')
        
        # 어노테이션 파일 로드
        self.landmarks = []
        self.image_paths = []
        
        with open(anno_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                parts = line.strip().split()
                # WFLW 포맷: 196개 좌표값 (98개 x,y쌍), 4개 bounding box값, 6개 속성, 이미지 경로
                if len(parts) >= 207:  # 최소 필요한 필드 수
                    # 98개 랜드마크 좌표 (x1, y1, x2, y2, ...)
                    coords = [float(p) for p in parts[:196]]
                    landmarks = np.array(coords).reshape(-1, 2)
                    
                    # 이미지 파일명
                    img_file = parts[-1]
                    img_path = os.path.join(self.image_dir, img_file)
                    
                    if os.path.exists(img_path):
                        self.landmarks.append(landmarks)
                        self.image_paths.append(img_path)
        
        print(f"{'훈련' if is_train else '테스트'} 데이터셋: {len(self.image_paths)}개 이미지 로드됨")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        landmarks = self.landmarks[idx].copy()
        
        # 이미지 로드
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"이미지 로드 오류: {img_path}, {e}")
            # 에러 발생 시 검은색 이미지로 대체
            image = Image.new('RGB', (200, 200), (0, 0, 0))
            landmarks = np.zeros((98, 2), dtype=np.float32)
        
        # 이미지 크기 정보
        w, h = image.size
        
        # 랜드마크 정규화 (0~1 범위로)
        landmarks = landmarks / np.array([w, h])
        
        # 변환 적용
        if self.transform:
            image = self.transform(image)
        
        return image, torch.tensor(landmarks, dtype=torch.float32)


# Spatial Configuration Network (SCN) 모델 정의
class SCN(nn.Module):
    def __init__(self, num_landmarks=98):
        super(SCN, self).__init__()
        
        # CNN 특징 추출기 - ResNet 스타일의 블록 추가
        self.features = nn.Sequential(
            # 초기 컨볼루션 레이어
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            
            # 레지듀얼 블록 1
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 레지듀얼 블록 2
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # 레지듀얼 블록 3
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # 전역 평균 풀링
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        # 공간 구성 모듈
        self.spatial_config = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_landmarks * 2)  # x, y 좌표
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        landmarks = self.spatial_config(x)
        
        # 랜드마크를 (batch, num_landmarks, 2) 형태로 reshape
        batch_size = landmarks.size(0)
        landmarks = landmarks.view(batch_size, -1, 2)
        
        return landmarks


# 가중치 초기화 함수
def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


# 훈련 함수
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=30):
    best_val_loss = float('inf')
    history = {'train_loss': [], 'val_loss': []}
    
    for epoch in range(num_epochs):
        # 훈련 모드
        model.train()
        epoch_loss = 0.0
        
        for images, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Training"):
            images = images.to(device)
            targets = targets.to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        # 에폭당 평균 훈련 손실
        avg_train_loss = epoch_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # 검증 모드
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in tqdm(val_loader, desc=f"Epoch {epoch+1}/{num_epochs} - Validation"):
                images = images.to(device)
                targets = targets.to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
        
        # 에폭당 평균 검증 손실
        avg_val_loss = val_loss / len(val_loader)
        history['val_loss'].append(avg_val_loss)
        
        # 스케줄러 업데이트 (있는 경우)
        if scheduler is not None:
            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
        
        # 진행 상황 출력
        print(f'Epoch {epoch+1}/{num_epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # 최고 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), 'best_scn_model.pth')
            print(f'Epoch {epoch+1}: 새로운 최고 모델 저장됨 (Val Loss: {best_val_loss:.4f})')
    
    return model, history


# 시각화 함수
def visualize_landmarks(image, landmarks, title=None):
    """이미지와 랜드마크 시각화"""
    # 이미지가 텐서면 numpy로 변환
    if isinstance(image, torch.Tensor):
        image = image.permute(1, 2, 0).numpy()
        # 역정규화
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        image = image * std + mean
        image = np.clip(image, 0, 1)
        
    plt.figure(figsize=(8, 8))
    plt.imshow(image)
    
    # 랜드마크 플롯
    landmarks = landmarks.detach().cpu().numpy()
    h, w = image.shape[:2]
    landmarks = landmarks * np.array([w, h])  # 원래 이미지 크기로 역정규화
    
    # 얼굴 부위별로 다른 색상 사용
    colors = ['r', 'g', 'b', 'c', 'm', 'y']
    
    # 98개 랜드마크 구역 나누기
    regions = [
        (0, 33),    # 얼굴 윤곽선 (빨강)
        (33, 51),   # 눈썹 (초록)
        (51, 67),   # 눈 (파랑)
        (67, 79),   # 코 (청록)
        (79, 91),   # 입 외곽 (마젠타)
        (91, 98)    # 입 내부 (노랑)
    ]
    
    for i, (start, end) in enumerate(regions):
        region_landmarks = landmarks[start:end]
        plt.scatter(region_landmarks[:, 0], region_landmarks[:, 1], c=colors[i], marker='o', s=20)
    
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


# 테스트 함수
def test_model(model, test_loader):
    model.eval()
    test_error = 0.0
    all_predictions = []
    all_targets = []
    all_images = []
    
    with torch.no_grad():
        for images, targets in tqdm(test_loader, desc="테스트 중"):
            images = images.to(device)
            targets = targets.to(device)
            
            outputs = model(images)
            
            # 첫 5개 샘플만 시각화용으로 저장
            if len(all_images) < 5:
                batch_size = min(5 - len(all_images), images.size(0))
                all_images.extend(images[:batch_size].cpu())
                all_predictions.extend(outputs[:batch_size].cpu())
                all_targets.extend(targets[:batch_size].cpu())
            
            # MSE 계산
            error = torch.nn.functional.mse_loss(outputs, targets).item()
            test_error += error
    
    # 평균 테스트 오차
    avg_test_error = test_error / len(test_loader)
    print(f'테스트 MSE: {avg_test_error:.4f}')
    
    # 결과 시각화
    for i in range(len(all_images)):
        image = all_images[i]
        # 이미지 역정규화
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        image = image * std + mean
        image = torch.clamp(image, 0, 1)
        
        # NumPy 배열로 변환
        image = image.permute(1, 2, 0).numpy()
        
        pred_landmarks = all_predictions[i]
        true_landmarks = all_targets[i]
        
        plt.figure(figsize=(12, 6))
        
        # 예측 랜드마크
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        h, w = image.shape[:2]
        pred_points = pred_landmarks.numpy() * np.array([w, h])
        
        # 얼굴 부위별로 다른 색상 사용
        colors = ['r', 'g', 'b', 'c', 'm', 'y']
        regions = [
            (0, 33),    # 얼굴 윤곽선 (빨강)
            (33, 51),   # 눈썹 (초록)
            (51, 67),   # 눈 (파랑)
            (67, 79),   # 코 (청록)
            (79, 91),   # 입 외곽 (마젠타)
            (91, 98)    # 입 내부 (노랑)
        ]
        
        for j, (start, end) in enumerate(regions):
            region_landmarks = pred_points[start:end]
            plt.scatter(region_landmarks[:, 0], region_landmarks[:, 1], c=colors[j], marker='o', s=20)
        
        plt.title('예측 랜드마크')
        plt.axis('off')
        
        # 실제 랜드마크
# 실제 랜드마크
        plt.subplot(1, 2, 2)
        plt.imshow(image)
        true_points = true_landmarks.numpy() * np.array([w, h])
        
        # 얼굴 부위별로 다른 색상 사용
        for j, (start, end) in enumerate(regions):
            region_landmarks = true_points[start:end]
            plt.scatter(region_landmarks[:, 0], region_landmarks[:, 1], c=colors[j], marker='o', s=20)
        
        plt.title('실제 랜드마크')
        plt.axis('off')
        
        plt.tight_layout()
        plt.show()
    
    return avg_test_error


# 메인 함수
def main():
    # 필요한 패키지 설치
    try:
        import gdown
    except ImportError:
        print("gdown 패키지 설치 중...")
    
    # 1. 데이터셋 다운로드
    success = download_wflw_dataset()
    if not success:
        print("데이터셋 준비에 실패했습니다. 프로그램을 종료합니다.")
        return
    
    # 2. 데이터 변환 정의
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 더 큰 입력 크기 사용
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 3. 데이터셋 로드
    train_dataset = WFLWDataset(root_dir='data/wflw', is_train=True, transform=transform)
    test_dataset = WFLWDataset(root_dir='data/wflw', is_train=False, transform=transform)
    
    # 데이터셋이 너무 작은 경우 처리
    if len(train_dataset) < 10:
        print("훈련 데이터셋이 너무 작습니다. 테스트용 더미 데이터셋을 다시 생성합니다.")
        create_dummy_dataset()
        train_dataset = WFLWDataset(root_dir='data/wflw', is_train=True, transform=transform)
        test_dataset = WFLWDataset(root_dir='data/wflw', is_train=False, transform=transform)
    
    # 4. 검증 데이터셋 분할 (훈련 데이터의 15%)
    train_size = int(len(train_dataset) * 0.85)
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    print(f"훈련 데이터 크기: {len(train_dataset)}")
    print(f"검증 데이터 크기: {len(val_dataset)}")
    print(f"테스트 데이터 크기: {len(test_dataset)}")
    
    # 5. 데이터 로더 생성
    batch_size = 32
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # 6. 랜드마크 수 확인 및 모델 초기화
    # 첫 번째 샘플을 통해 랜드마크 수 확인
    _, landmarks = train_dataset[0]
    num_landmarks = landmarks.shape[0]
    print(f"랜드마크 수: {num_landmarks}")
    
    model = SCN(num_landmarks=num_landmarks)
    model.apply(init_weights)  # 가중치 초기화
    model.to(device)
    
    # 7. 손실 함수 및 옵티마이저 정의
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    
    # 학습률 스케줄러 추가
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # 8. 모델 훈련
    print("모델 훈련 시작...")
    epochs = 30
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        num_epochs=epochs
    )
    
    # 9. 훈련 결과 시각화
    plt.figure(figsize=(10, 5))
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    # 10. 베스트 모델 로드
    model.load_state_dict(torch.load('best_scn_model.pth'))
    
    # 11. 테스트셋으로 평가
    print("테스트셋으로 모델 평가 중...")
    test_error = test_model(model, test_loader)
    
    print("훈련 완료!")
    print(f"최종 테스트 MSE: {test_error:.4f}")
    
    # 12. 모델 저장
    model_info = {
        'state_dict': model.state_dict(),
        'num_landmarks': num_landmarks,
        'test_error': test_error,
        'train_history': history
    }
    torch.save(model_info, 'scn_model_wflw_complete.pth')
    print("모델 저장 완료: scn_model_wflw_complete.pth")


if __name__ == "__main__":
    main()
