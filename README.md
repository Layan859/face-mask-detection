#Face Detection Mask

This project implements a face mask detection model capable of identifying three classes:
with_mask
without_mask
mask_worn_incorrectly

The system uses YOLOv5 and a publicly available labeled dataset. The dataset was converted to YOLO format and split into train/val/test sets using a custom preprocessing script.

1. Project Structure
prepare_dataset.py — converts XML annotations to YOLO format and splits the dataset
mask.yaml — YOLOv5 dataset configuration file
best.pt — final trained YOLOv5 model weights
README.md — project documentation and setup instructions
Face Mask Detection Report.docx — full written report
results/ — folder containing training outputs and prediction examples:
F1_curve.png
PR_curve.png
confusion_matrix.png
labels.jpg
labels_correlogram.jpg
results.png
sample_with_mask.png
sample_without_mask.png
sample_crowd.png
datasets/ (not uploaded to GitHub) — local dataset used for training

2. Dataset Download & Preparation
Dataset Source
This project uses the Face Mask Detection dataset available from Kaggle:
🔗 https://www.kaggle.com/datasets/andrewmvd/face-mask-detection

How to Download
Download the dataset ZIP file from Kaggle.
Extract it into a folder named datasets/mask/
datasets/mask/
    ├── images/
    ├── annotations/

Convert Annotations to YOLO Format

Run:
python prepare_dataset.py

This script will:

Convert XML annotations → YOLO .txt format

Create 853 label files

Automatically split the data into train/val/test

Move images + labels into YOLO structure

3. Setup Instructions
Install Requirements
pip install torch torchvision
pip install ultralytics
pip install pandas matplotlib seaborn

Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

4. Training the Model

Run YOLOv5 training:

python train.py --img 640 --batch 16 --epochs 30 --data ../mask.yaml --weights yolov5s.pt --name mask_detector2

5. Reproducing Key Results

After training finishes, YOLOv5 automatically generates:

Confusion matrix

PR curve

F1 curve

Loss curves

Label distribution images

All results can be viewed inside:

runs/train/mask_detector2/


Example outputs are saved in this repository under the results/ folder.

6. Running Inference (Testing)

To test the model:

python detect.py --weights best.pt --img 640 --source test_images/


This produces detection images like:

sample_with_mask.png

sample_without_mask.png

sample_crowd.png

Saved inside:

runs/detect/exp/

7. Model Performance Summary
Metric	Score
mAP (50)	~0.68
Precision	~0.52
Recall	~0.69
