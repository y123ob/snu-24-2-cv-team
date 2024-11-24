# StairMap: Automated Accessibility Mapping via Delivery Workers' Body Camera Analysis

## SNU CV 2024-Fall Team Project

This repository contains the implementation and research materials for the
project **"StairMap: Automated Accessibility Mapping via Delivery Workers' Body
Camera Analysis"**, conducted as part of the **Computer Vision course** at Seoul
National University in Fall 2024.

---

## Project Overview

### Background

Ensuring unrestricted mobility access for people with disabilities in urban
environments is a critical social challenge. Architectural barriers, such as
stairs in building entrances or indoor spaces, significantly hinder mobility for
wheelchair users and elderly individuals. In South Korea, where the number of
mobility-impaired individuals reached 15.51 million as of 2021, the lack of
updated accessibility information creates further obstacles.

### Problem Statement

Currently, mobility-impaired individuals rely on outdated or incomplete
information, such as blog reviews or static street views, which fail to provide
essential details about indoor pathways. Physical surveys by volunteers are
unsustainable for large-scale updates due to the vast number of establishments.

### Proposed Solution

This project aims to develop a computer vision-based system that leverages
delivery workers' body camera footage to analyze and extract accessibility
information. The proposed system will:

1. Automatically detect stairs and other obstacles in pathways.
2. Provide scalable and regularly updated accessibility data.
3. Address gaps in indoor pathway information, invisible to traditional methods.

---

## Project Goals

- Develop a scalable, automated pipeline for analyzing body camera footage.
- Implement state-of-the-art computer vision techniques to detect stairs and
  classify accessibility features.
- Demonstrate the feasibility of integrating this system into existing delivery
  infrastructures.

---

## Team Members

- **Yooseok Jeong**: y123ob@snu.ac.kr
- **Jongbum Won**: jim8697@snu.ac.kr
- **Harim Jang**: jhr337@snu.ac.kr
- **Minwoo Lee**: leemm011@snu.ac.kr

---

## Tech Stack

- **Programming Language**: Python
- **Frameworks**: PyTorch, TensorFlow, OpenCV

---

## Dataset

- **Dataset**:
  [Kaggle Dataset](https://www.kaggle.com/datasets/nderalparslan/dwsonder/data)

  This dataset consists of images from three different datasets and other
  images gathered from the Internet. Objects such as doors, windows, and stairs
  are labeled using the YOLO annotation tool, with labels stored in text files:
  - `0`: Door
  - `1`: Stairs
  - `2`: Window
- **Custom Video Data**: Body camera footage captured and processed specifically
  for this project to supplement the primary dataset and test real-world
  conditions.

---

## Experimental Methods

1. **Traditional CV Techniques**: Edge detection (Canny, Sobel) and Hough
   Transform.
2. **Deep Learning Models**: Pre-trained object detection models like YOLO and
   Resnet.

---

## Usage

### Step 1: Dataset Preparation
1. Download the dataset from [Kaggle Dataset](https://www.kaggle.com/datasets/nderalparslan/dwsonder/data).
2. Extract the dataset into a folder, for example:
   ```
   data/kaggle/
   ```
   Ensure the folder contains the necessary images and annotation files.

### Step 2: Preprocessing the Dataset
Run the preprocessing script to reorganize and prepare the dataset for training and testing:
```bash
python preprocess.py --input data/kaggle --output data/processed
```
This will generate a structured folder with preprocessed images and labels in `data/processed`.

---

### Step 3: Running the Deep Learning Module
To train the deep learning model, use the following command:
```bash
python train.py --config configs/deep_learning.yaml #TODO: 
```
Ensure the configuration file includes paths to the processed dataset.

To test the trained deep learning model:
```bash
python test.py --model outputs/deep_model.pth --input data/processed/test #TODO:
```

---

### Step 4: Testing Traditional Methods
To test stair detection using traditional computer vision techniques:
```bash
python traditional_methods.py --input data/processed/test #TODO:
```

---

### Step 5: Running Real Video Tests
You can test both deep learning and traditional methods on real video data. Place your test videos in the following directory:
```
data/videos/
```

To run the test:
```bash
python real_video_test.py --video data/videos/sample.mp4 --method [deep|traditional|both] #TODO:
```

- Use `--method deep` to apply the deep learning model.
- Use `--method traditional` for traditional methods.
- Use `--method both` to compare the two approaches.



> **Note:** This README will be updated as the project progresses.
