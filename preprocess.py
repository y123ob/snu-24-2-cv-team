import os
import shutil

# 경로 설정
kaggle_dir = "data/kaggle"  # Kaggle 데이터 경로
output_dir = "data/processed"  # 전처리된 데이터 저장 경로


# 디렉토리 생성
def ensure_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


# 전처리 데이터를 저장할 디렉토리 구성
no_stairs_dir = os.path.join(output_dir, "no_stairs")  # 계단 없음
stairs_dir = os.path.join(output_dir, "stairs")  # 계단 있음

for dir_path in [output_dir, no_stairs_dir, stairs_dir]:
    ensure_dir(dir_path)

no_stairs_count = 1
stairs_count = 1

jpg_images = 0

# 이미지 및 레이블 처리
for filename in os.listdir(kaggle_dir):
    if filename.endswith(".jpg"):
        jpg_images += 1
        image_path = os.path.join(kaggle_dir, filename)
        label_path = os.path.join(kaggle_dir, filename.replace(".jpg", ".txt"))

        if os.path.exists(label_path):
            if os.path.getsize(label_path) == 0:
                label = 0  # 계단 없음
            else:
                with open(label_path, "r") as f:
                    content = f.read()
                    class_numbers = [
                        line.split()[0]
                        for line in content.strip().split("\n")
                        if line.strip()
                    ]
                    label = 1 if "1" in class_numbers else 0  # 클래스 '1'은 계단
        else:
            label = 0

        # 이미지 이름 변경 및 이동
        if label == 0:
            new_name = f"no_stairs_{no_stairs_count}.jpg"
            no_stairs_count += 1
            shutil.copy(image_path, os.path.join(no_stairs_dir, new_name))
        else:
            new_name = f"stairs_{stairs_count}.jpg"
            stairs_count += 1
            shutil.copy(image_path, os.path.join(stairs_dir, new_name))

print(f"Preprocessed data saved to {output_dir}")
print(f"Total images: {jpg_images}")
