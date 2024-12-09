import os
import cv2
import pandas as pd
from deep_learning_methods import resnet_detect
from traditional_methods import trad_detect
from sklearn.metrics import accuracy_score
from tqdm import tqdm  # tqdm 추가

# 이미지 파일 경로
no_stairs_folder = "data/processed/no_stairs"
stairs_folder = "data/processed/stairs"

# 실제 레이블 (1: stairs 있음, 0: stairs 없음)
labels = []

# 예측 결과 저장
pred_dl = []
pred_trad = []

# 'no_stairs' 폴더에서 이미지 로드
for i in tqdm(range(1, 3497), desc="Processing 'no_stairs' images", unit="image"):
    image_path = os.path.join(no_stairs_folder, f"no_stairs_{i}.jpg")
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        # 레이블은 0 (stairs 없음)
        labels.append(0)
        pred_dl.append(round(resnet_detect(img)))
        pred_trad.append(round(trad_detect(img)))

# 'stairs' 폴더에서 이미지 로드
for i in tqdm(range(1, 1517), desc="Processing 'stairs' images", unit="image"):
    image_path = os.path.join(stairs_folder, f"stairs_{i}.jpg")
    if os.path.exists(image_path):
        img = cv2.imread(image_path)
        # 레이블은 1 (stairs 있음)
        labels.append(1)
        pred_dl.append(round(resnet_detect(img)))
        pred_trad.append(round(trad_detect(img)))

# 정확도 계산
accuracy_dl = accuracy_score(labels, pred_dl)
accuracy_trad = accuracy_score(labels, pred_trad)

print(f"Accuracy of dl: {accuracy_dl * 100:.2f}%")
print(f"Accuracy of trad: {accuracy_trad * 100:.2f}%")

# 결과를 데이터프레임으로 저장
results_df = pd.DataFrame({"label": labels, "pred_dl": pred_dl, "pred_trad": pred_trad})

# 결과를 CSV 파일로 저장
results_df.to_csv("prediction_results.csv", index=False)

print("Prediction results saved to 'prediction_results.csv'.")
