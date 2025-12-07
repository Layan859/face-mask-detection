import os
import xml.etree.ElementTree as ET
import random
import shutil

# === PATHS (automatic, don't change) ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ANN_DIR = os.path.join(BASE_DIR, "annotations")
IMG_DIR = os.path.join(BASE_DIR, "images")
DATASET_DIR = os.path.join(BASE_DIR, "datasets", "mask")

# temp folder to store all YOLO label files before splitting
ALL_LABELS_DIR = os.path.join(BASE_DIR, "labels_all")
os.makedirs(ALL_LABELS_DIR, exist_ok=True)

# class mapping
CLASS_MAP = {
    "with_mask": 0,
    "without_mask": 1,
    "mask_weared_incorrect": 2
}

def convert_xml_to_yolo():
    print("Converting XML annotations to YOLO format...")
    count = 0
    image_info_list = []  # (image_filename, label_filename_base)

    for xml_name in os.listdir(ANN_DIR):
        if not xml_name.endswith(".xml"):
            continue

        xml_path = os.path.join(ANN_DIR, xml_name)
        tree = ET.parse(xml_path)
        root = tree.getroot()

        # image filename and size
        img_filename = root.find("filename").text
        size = root.find("size")
        img_w = float(size.find("width").text)
        img_h = float(size.find("height").text)

        # label file base (no extension)
        base_name = os.path.splitext(img_filename)[0]
        label_path = os.path.join(ALL_LABELS_DIR, base_name + ".txt")

        lines = []
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in CLASS_MAP:
                continue
            class_id = CLASS_MAP[class_name]

            bndbox = obj.find("bndbox")
            xmin = float(bndbox.find("xmin").text)
            ymin = float(bndbox.find("ymin").text)
            xmax = float(bndbox.find("xmax").text)
            ymax = float(bndbox.find("ymax").text)

            # Convert to YOLO format (normalized)
            x_center = ((xmin + xmax) / 2.0) / img_w
            y_center = ((ymin + ymax) / 2.0) / img_h
            w = (xmax - xmin) / img_w
            h = (ymax - ymin) / img_h

            line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
            lines.append(line)

        if not lines:
            continue

        with open(label_path, "w") as f:
            f.write("\n".join(lines))

        image_info_list.append(img_filename)
        count += 1

    print(f"Done. Created {count} label files.")
    return image_info_list


def find_image_path(filename):
    """Return full path to image (png/jpg/jpeg) given a filename."""
    base = os.path.splitext(filename)[0]
    for ext in [".png", ".jpg", ".jpeg"]:
        path = os.path.join(IMG_DIR, base + ext)
        if os.path.exists(path):
            return path
    return None


def split_and_move_files(image_filenames, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    print("Splitting into train/val/test and copying files...")

    # ensure ratios sum to 1
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6

    random.shuffle(image_filenames)
    n = len(image_filenames)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    # rest go to test
    n_test = n - n_train - n_val

    train_files = image_filenames[:n_train]
    val_files = image_filenames[n_train:n_train + n_val]
    test_files = image_filenames[n_train + n_val:]

    splits = {
        "train": train_files,
        "val": val_files,
        "test": test_files
    }

    for split_name, files in splits.items():
        img_out_dir = os.path.join(DATASET_DIR, "images", split_name)
        lbl_out_dir = os.path.join(DATASET_DIR, "labels", split_name)
        os.makedirs(img_out_dir, exist_ok=True)
        os.makedirs(lbl_out_dir, exist_ok=True)

        moved = 0
        for img_filename in files:
            base = os.path.splitext(img_filename)[0]
            img_src = find_image_path(img_filename)
            if img_src is None:
                continue

            lbl_src = os.path.join(ALL_LABELS_DIR, base + ".txt")
            if not os.path.exists(lbl_src):
                continue

            img_dst = os.path.join(img_out_dir, os.path.basename(img_src))
            lbl_dst = os.path.join(lbl_out_dir, os.path.basename(lbl_src))

            shutil.copy2(img_src, img_dst)
            shutil.copy2(lbl_src, lbl_dst)
            moved += 1

        print(f"{split_name}: moved {moved} images + labels")

    print("Done splitting and copying.")


if __name__ == "__main__":
    imgs = convert_xml_to_yolo()
    split_and_move_files(imgs)
    print("All done! Check datasets/mask/images and datasets/mask/labels.")