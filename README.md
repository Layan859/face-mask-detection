# face-mask-detection
This project implements a face mask detection system capable of identifying three classes:
with_mask, without_mask, and mask_worn_incorrectly.
A custom dataset was preprocessed, converted to YOLO format, and trained using YOLOv5.

1. Project Overview
The goal of this project is to develop a real-time model that can detect whether individuals are wearing face masks correctly. Applications include public safety monitoring, access control systems, and automated compliance checking.
The model was trained and evaluated using YOLOv5.
Training outputs such as confusion matrix, PR/F1 curves, and sample predictions are included in the repository.

2. Repository Structure
face-mask-detection/
│
├── prepare_dataset.py          # Converts XML annotations to YOLO format and splits dataset
├── mask.yaml                   # Dataset configuration file for YOLOv5
├── best.pt                     # Final trained model weights
│
├── results/                    # Training results and evaluation plots
│   ├── F1_curve.png
│   ├── PR_curve.png
│   ├── confusion_matrix.png
│   ├── labels.jpg
│   ├── labels_correlogram.jpg
│   ├── results.png
│   ├── sample crowd.png
│   ├── sample with mask.png
│   ├── sample without mask.png
│   ├── train_batch0.jpg
│   ├── train_batch1.jpg
│   └── placeholder.txt         # Enables GitHub folder visibility
│
├── Face Mask Detection Report.docx   # Full project report
└── README.md                   # Project documentation (this file)

3. Setup Instructions
Install Dependencies
pip install torch torchvision
pip install pandas matplotlib seaborn
pip install ultralytics

Clone YOLOv5
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

4. Dataset Preparation
XML annotations were converted to YOLO format using:
python prepare_dataset.py

This script performs:
XML → YOLO text label conversion
Dataset splitting into train / val / test
Automatic folder organization

The dataset contains 3 classes:
with_mask
without_mask
mask_worn_incorrectly

5. Training the Model

Training was performed with:
python train.py --img 640 --batch 16 --epochs 30 --data mask.yaml --weights yolov5s.pt

Output files include:
best.pt – best performing model
Confusion matrix
F1 score curve
Precision-Recall curve
Train/validation batches

6. Sample Predictions
Examples of model predictions are included in:

results/
 ├── sample with mask.png
 ├── sample without mask.png
 └── sample crowd.png


These demonstrate the model detecting multiple classes with confidence scores.

7. Results Summary
Key evaluation metrics:
Metric	Value
Precision (overall)	~0.52
Recall (overall)	~0.69
mAP@0.5	~0.69
mAP@0.5:0.95	~0.44

These values are based on the final validation results during training.

8. How to Use the Model

To run inference:
python detect.py --weights best.pt --img 640 --source your_image.jpg
or for real-time camera input:
python detect.py --weights best.pt --source 0

The model will output bounding boxes and class predictions.

9. Acknowledgements

YOlOv5 by Ultralytics
Dataset sourced from open mask detection datasets
Implementation adapted for academic use
