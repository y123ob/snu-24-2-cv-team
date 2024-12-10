import cv2
import numpy as np
from scipy import ndimage

def create_gabor_filter(ksize, sigma, theta, lambd, gamma):
    """Gabor 필터 생성"""
    kernel = cv2.getGaborKernel((ksize, ksize), sigma, theta, lambd, gamma, 0, ktype=cv2.CV_32F)
    return kernel / kernel.sum()  # 정규화 추가

def detect_stairs(image):
    """계단 검출을 위한 Gabor 필터 적용"""
    # 이미지 전처리
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5,5), 0)  # 노이즈 제거
    
    # Gabor 필터 파라미터
    ksize = 10  # 크기 감소
    sigma = 3
    theta_values = [np.pi, np.pi/2, np.pi/6, np.pi/4, np.pi/3, 2*np.pi/3, 3*np.pi/4, 5*np.pi/6]  # 더 많은 각도
    lambd = 15
    gamma = 0.5
    
    filtered_images = []
    
    # 여러 방향의 Gabor 필터 적용
    for theta in theta_values:
        kernel = create_gabor_filter(ksize, sigma, theta, lambd, gamma)
        filtered = cv2.filter2D(gray, cv2.CV_8U, kernel)  # CV_8UC3를 CV_8U로 수정
        filtered_images.append(filtered)
    
    # 필터링된 이미지 결합
    combined = np.maximum.reduce(filtered_images)
    
    # 적응형 이진화
    binary = cv2.adaptiveThreshold(
        combined,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11,
        2
    )
    
    # 노이즈 제거
    kernel = np.ones((5,5), np.uint8)
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel)
    
    return cleaned

def process_image(image_path):
    """이미지 처리 및 계단 검출"""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("이미지를 불러올 수 없습니다.")
    
    # 계단 검출
    result = detect_stairs(image)
    
    # 컨투어 필터링 추가
    contours, _ = cv2.findContours(result, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    filtered_contours = []
    
    # 면적 기반 필터링
    min_area = image.shape[0] * image.shape[1] * 0.01  # 이미지 크기의 1%
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > min_area:
            filtered_contours.append(contour)
    
    # 결과 시각화
    overlay = image.copy()
    cv2.drawContours(overlay, filtered_contours, -1, (0, 255, 0), 2)
    
    return overlay, result  # 이진화 결과도 반환

# 사용 예시
if __name__ == "__main__":
    image_path = "stairs.jpg"
    overlay, binary_result = process_image(image_path)
    
    # 결과 표시
    cv2.imshow("Original with Detections", overlay)
    cv2.imshow("Binary Result", binary_result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()