import cv2  # OpenCV로 비디오 및 이미지 처리
import os
import torch  # PyTorch 딥러닝 프레임워크
import torchvision.transforms as transforms  # 이미지 전처리용 transform 모듈
from torchvision import models  # 사전 학습된 모델 로드
from collections import deque  # 최근 감지 상태를 트래킹하기 위한 deque

model = None
transform = None
device = None


def resnet_init():
    global model, transform, device
    # Load the pre-trained model
    model_path = "./DeepLearning/model/resnet18_staircase.pth"

    if not os.path.exists(model_path):
        print(f"Error: Model file {model_path} does not exist.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18()
    model.fc = torch.nn.Linear(
        model.fc.in_features, 1
    )  # 1 class: stair (binary classification)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    # Define image transformation
    transform = transforms.Compose(
        [
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )


def resnet_detect(frame) -> float:
    if model is None or transform is None or device is None:
        resnet_init()

    input_tensor = transform(frame).unsqueeze(0).to(device)
    # Preprocess the frame
    input_tensor = transform(frame).unsqueeze(0).to(device)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.sigmoid(output).item()  # Sigmoid for binary classification
        return prediction
