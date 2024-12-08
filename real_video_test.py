import argparse  # 명령줄 인자 파싱
import os  # 파일 경로 확인 및 운영체제 기능
import cv2  # OpenCV로 비디오 및 이미지 처리
from collections import deque  # 최근 감지 상태를 트래킹하기 위한 deque
from typing import Callable, Any  # 함수 타입 힌트용
import deep_learning_methods


def deep_learning_method(frame):
    """Detect stairs in a video using a pre-trained deep learning model."""
    return deep_learning_methods.resnet_detect(frame)


def traditional_method(frame):
    """Placeholder function for traditional computer vision-based testing."""
    # TODO: Implement the traditional computer vision inference here


def combined_method(frame):
    """Run both deep learning and traditional methods for testing."""
    # TODO: Implement the combined method here


# Detect stairs in a video using the specified detection method
#
# This function processes a video file to detect the presence of stairs using a user-provided detection method.
# The detection method should analyze the video and return a confidence score (float between 0 and 1)
# indicating the likelihood of stairs being present in the video. Based on the confidence score,
# this function determines if stairs are detected and returns True or False.
#
# Parameters:
# - video_path (str): The file path of the video to be analyzed.
# - detect_method (Callable[[str], float]): A function that takes the video file path as input and returns a
#   confidence score (float between 0 and 1). The score represents the predicted probability of stairs being present
#   in the video.
#
# Returns:
# - bool:
#   - True If stairs are detected in the video. else, False.
def detect_stairs(video_path, detect_method: Callable[[Any], float]) -> bool:
    # Open the video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}.")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get frames per second
    frame_interval = fps  # 1 second interval
    frame_count = 0

    stair_detections = deque(maxlen=3)  # Track the last 3 detections
    stair_detected = False

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:  # Process every 1 second
            prediction = detect_method(frame)
            label = 1 if prediction >= 0.5 else 0

            # Print detection result for the current frame
            if label == 1:
                print(f"Frame {frame_count // frame_interval}: Stairs detected")
            else:
                print(f"Frame {frame_count // frame_interval}: No stairs detected")

            # Update stair detection queue
            stair_detections.append(label)

            # Check if stairs are consistently detected
            if stair_detections.count(1) >= 3:  # 3 consecutive "stair" detections
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
        detect_stairs(video_path, deep_learning_method)
    elif method == "traditional":
        detect_stairs(video_path, traditional_method)
    elif method == "both":
        detect_stairs(video_path, combined_method)


if __name__ == "__main__":
    main()
