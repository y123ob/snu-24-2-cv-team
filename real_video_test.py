import argparse  # 명령줄 인자 파싱
import os  # 파일 경로 확인 및 운영체제 기능
import cv2  # OpenCV로 비디오 및 이미지 처리
import torch  # PyTorch 딥러닝 프레임워크
import torchvision.transforms as transforms  # 이미지 전처리용 transform 모듈
from torchvision import models  # 사전 학습된 모델 로드
from collections import deque  # 최근 감지 상태를 트래킹하기 위한 deque
import deep_learning_methods


def deep_learning_method(video_path):
    """Detect stairs in a video using a pre-trained deep learning model."""
    print(f"Running deep learning method on {video_path}")

    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_interval = fps  # 1 second interval
    frame_count = 0

    stair_detections = deque(maxlen=3)  # Track the last 5 detections
    stair_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Process every 1 second
            prediction = deep_learning_methods.resnet_detect(frame)
            label = 1 if prediction >= 0.5 else 0

            # Print detection result for the current frame
            if label == 1:
                print(f"Frame {frame_count // frame_interval}: Stairs detected")
            else:
                print(f"Frame {frame_count // frame_interval}: No stairs detected")

            # Update stair detection queue
            stair_detections.append(label)

            # Check if stairs are consistently detected
            if stair_detections.count(1) >= 3:  # 5 consecutive "stair" detections
                stair_detected = True
                break

        frame_count += 1

    cap.release()

    if stair_detected:
        print(f"Stairs detected in the video: {video_path}")
        return True
    else:
        print(f"No stairs detected in the video: {video_path}")
        return False


def traditional_method(video_path):
    """Placeholder function for traditional computer vision-based testing."""
    print(f"Running traditional method on {video_path}")
    # TODO: Implement the traditional computer vision inference here


def combined_method(video_path):
    """Run both deep learning and traditional methods for testing."""
    print(f"Running both methods on {video_path}")
    # TODO: Implement the combined method here


def main():
    parser = argparse.ArgumentParser(description="Test a video using selected methods.")
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the video file to be tested.",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["deep", "traditional", "both"],
        required=True,
        help="The method to use for testing: 'deep', 'traditional', or 'both'.",
    )

    args = parser.parse_args()

    video_path = args.video
    method = args.method

    if not os.path.exists(video_path):
        print(f"Error: Video file {video_path} does not exist.")
        return

    if method == "deep":
        deep_learning_method(video_path)
    elif method == "traditional":
        traditional_method(video_path)
    elif method == "both":
        combined_method(video_path)


if __name__ == "__main__":
    main()
