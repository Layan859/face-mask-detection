# face-mask-detection

1. Project Overview
This project implements an object detection model capable of identifying three classes of face mask usage:
with_mask
without_mask
mask_weared_incorrect

The model is trained on a dataset of annotated images, converted from XML (Pascal VOC format) to YOLO format using a custom preprocessing script. Training was performed using YOLOv5.

<pre> face-mask-detection/ ├── prepare_dataset.py # Script for converting XML to YOLO format and splitting dataset ├── mask.yaml # Dataset configuration file for YOLOv5 ├── best.pt # Final trained model weights ├── results.png # Summary of training results ├── confusion_matrix.png # Confusion matrix ├── PR_curve.png # Precision–Recall curve ├── F1_curve.png # F1-score curve ├── labels.jpg # Label distribution visualization ├── labels_correlogram.jpg # Label correlation plot ├── train_batch0.jpg # Training batch sample ├── with mask.png # Sample prediction (with mask) ├── without mask.png # Sample prediction (without mask) ├── with and without mask.png # Sample prediction (mixed) ├── Face Mask Detection Report.docx # Full project report └── README.md # Project documentation </pre>

3. Setup Instructions
Install Dependencies

This project requires Python 3.8+.

Run the following commands:
pip install torch torchvision
pip install pandas matplotlib seaborn
pip install ultralytics


Clone YOLOv5:
git clone https://github.com/ultralytics/yolov5
cd yolov5
pip install -r requirements.txt

4. Dataset Preparation

Place the following folders in the project directory:
annotations/   # XML annotation files
images/        # Raw images


Run the dataset preparation script:
python prepare_dataset.py


This script performs the following actions:

Converts XML annotations to YOLO format

Creates 853 label files

Splits data into train/val/test

Builds the directory structure:

datasets/mask/images/train/
datasets/mask/images/val/
datasets/mask/labels/train/
datasets/mask/labels/val/

5. Training the Model

Run YOLOv5 training:

python train.py --img 640 --batch 16 --epochs 30 --data mask.yaml --weights yolov5s.pt --project runs/train --name mask_detector

After training, the following files are generated:

best.pt (best-performing weights)

Training graphs and metrics

Confusion matrix and curve plots

6. Running Inference (Detection)

To test the model on a single image:

python detect.py --weights best.pt --img 640 --source path/to/image.jpg

YOLOv5 saves prediction outputs to:

runs/detect/exp/

Sample prediction images are included in this repository.

7. Results Summary

The model achieved the following:

High detection accuracy across the three mask classes

Clear visual predictions on crowded scenes

Strong performance shown through PR curve, F1 curve, and confusion matrix

See the training plots (PR_curve.png, F1_curve.png, confusion_matrix.png) for details.

8. Acknowledgements

This project uses the YOLOv5 architecture provided by Ultralytics.
Dataset conversion and training pipeline were implemented as part of a course project.
