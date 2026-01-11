
# LEGO Piece Finder – YOLO Dataset & Training

This repository contains the dataset preparation pipeline and training setup for the **LEGO Piece Finder**, an AI-based image recognition system that identifies LEGO pieces using YOLOv8.

The focus of this repository is reproducibility, clarity, and ease of use for future developers.

---

## Requirements

- Python 3.9+
- pip
- Jupyter Notebook (optional, for experimentation)

### Python Dependencies

```bash
pip install ultralytics opencv-python pandas scikit-learn torch torchvision
```

> A CUDA-enabled GPU is recommended.  
---

### Dataset Source (Manual Download)

The original LEGO brick image dataset must be downloaded manually from Kaggle:

Kaggle Dataset – LEGO Brick Sorting Image Recognition
https://www.kaggle.com/datasets/pacogarciam3/lego-brick-sorting-image-recognition

After downloading and extracting the dataset, place it in a local folder (for example lego_dataset/) and ensure it matches the expected structure described below.

> A Kaggle account is required to download the dataset.

## Folder Structure

### Input Dataset Structure

The original LEGO dataset is expected to be structured as follows:

```
lego_dataset/
├── cropped_images/
│   ├── class_name_1/
│   │   ├── image1.png
│   │   └── image2.png
│   ├── class_name_2/
│   └── ...
├── ImageSetKey.csv
```

- `cropped_images/`: Contains cropped LEGO part images grouped by class
- `ImageSetKey.csv`: Maps image names to class labels

---

### YOLO Output Structure (Auto-generated)

After running the preparation script:

```
lego_yolo/
├── images/
│   ├── train/
│   └── val/
├── labels/
│   ├── train/
│   └── val/
├── data.yaml
```

---

## Configuration Paths

The following paths must be set correctly before running the script:

```python
DATASET_PATH = "path/to/lego_dataset"
CROPPED_PATH = os.path.join(DATASET_PATH, "cropped_images")
CSV_PATH = os.path.join(DATASET_PATH, "ImageSetKey.csv")
OUT_PATH = "path/to/lego_yolo"
```

Make sure that:
- `cropped_images` exists inside `DATASET_PATH`
- `ImageSetKey.csv` exists in `DATASET_PATH`
- `OUT_PATH` points to an empty or new folder for YOLO output

---

## Dataset Preparation

The dataset preparation script performs the following steps:

1. Reads image–class mappings from `ImageSetKey.csv`
2. Splits the dataset into train (80%) and validation (20%) sets
3. Converts images into YOLO-compatible format
4. Generates bounding box labels (full-image bounding box per piece)
5. Creates a `data.yaml` file for YOLO training

Once completed, the dataset is ready for training.

---

## Model Training

YOLOv8 is used for training the LEGO Piece Finder detection model.

### Training Script

```python
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLOv8 nano detection model

model.train(
    data=r"C:/Users/Yannick/Documents/lego_yolo/data.yaml",
    epochs=50,
    imgsz=80,
    batch=16,
    project=r"C:/Users/Yannick/Documents/lego_yolo",
    name="lego_detection"
)
```

### Training Notes

- `yolov8n.pt` is used for fast experimentation
- Image size is kept small (80x80) due to cropped LEGO parts
- Batch size may be adjusted depending on GPU memory
- Training results are stored in:

```
lego_yolo/lego_detection/
```

This folder contains:
- Trained model weights
- Training logs
- Confusion matrices
- Precision and recall metrics

---

## Evaluation

Model performance is evaluated using:
- Confusion matrices
- Macro precision
- Top-1 and Top-3 accuracy

These metrics help identify misclassification patterns and guide further improvements.

---

## Notes for Future Developers

- Current setup supports **single LEGO piece per image**
- Multi-piece detection requires dataset expansion and label updates
- Alternative models (e.g. EfficientNet, Vision Transformers) may improve accuracy
- Dataset quality has a larger impact than model size

---

## License & Disclaimer

This project is developed for educational purposes at **Fontys ICT**.  
LEGO® is a trademark of the LEGO Group, which does not sponsor or endorse this project.
