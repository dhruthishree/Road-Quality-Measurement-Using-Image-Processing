"""
Script to extract and prepare the balanced Indian Road Damage Dataset (RDD2020)
from Hugging Face cache into train and test directories with YOLO labels.
"""

import os
import io
import shutil
import random
import zipfile
import xml.etree.ElementTree as ET

def extract_and_prepare_dataset(
    dest_dir="dataset/indian_roads_dataset",
    max_train_samples=2500,
    max_test_samples=600
):
    print("=" * 75)
    print("PREPARING REAL INDIAN ROAD DATASET (RDD2020 FROM HUGGING FACE)")
    print("=" * 75)

    train_zip = r"C:\Users\dhrut\.cache\huggingface\hub\datasets--ShixuanAn--RDD2020\snapshots\9e577db7f4e2eda6042c967545ce144791f4c305\train.zip"
    if not os.path.exists(train_zip):
        raise FileNotFoundError(f"train.zip not found at {train_zip}")

    train_out = os.path.join(dest_dir, "train")
    test_out = os.path.join(dest_dir, "test")
    os.makedirs(train_out, exist_ok=True)
    os.makedirs(test_out, exist_ok=True)

    print("[*] Opening nested India.zip from Hugging Face snapshot...")
    with zipfile.ZipFile(train_zip, "r") as z:
        india_bytes = z.read("train/India.zip")

    print("[*] Parsing Indian road annotations and images...")
    with zipfile.ZipFile(io.BytesIO(india_bytes)) as iz:
        all_files = set(iz.namelist())
        xml_files = [f for f in iz.namelist() if f.endswith(".xml") and not "._" in f]
        
        damaged_items = []
        clean_items = []

        for xf in xml_files:
            base_id = os.path.splitext(os.path.basename(xf))[0]
            # Corresponding image
            img_candidates = [
                f"India/images/{base_id}.jpg",
                f"images/{base_id}.jpg",
                f"{base_id}.jpg"
            ]
            img_path_in_zip = None
            for cand in img_candidates:
                if cand in all_files:
                    img_path_in_zip = cand
                    break

            if img_path_in_zip is None:
                continue

            # Read XML
            try:
                content = iz.read(xf).decode("utf-8", errors="ignore")
                root = ET.fromstring(content)
                objs = root.findall("object")
                
                boxes = []
                # Size
                sz = root.find("size")
                img_w = float(sz.find("width").text) if sz is not None and sz.find("width") is not None else 720.0
                img_h = float(sz.find("height").text) if sz is not None and sz.find("height") is not None else 720.0

                for o in objs:
                    name_tag = o.find("name")
                    bnd = o.find("bndbox")
                    if bnd is not None and name_tag is not None:
                        xmin = float(bnd.find("xmin").text)
                        ymin = float(bnd.find("ymin").text)
                        xmax = float(bnd.find("xmax").text)
                        ymax = float(bnd.find("ymax").text)
                        
                        cx = ((xmin + xmax) / 2.0) / img_w
                        cy = ((ymin + ymax) / 2.0) / img_h
                        w = abs(xmax - xmin) / img_w
                        h = abs(ymax - ymin) / img_h
                        boxes.append((name_tag.text.strip(), cx, cy, w, h))

                item_record = {
                    "id": base_id,
                    "img_path_in_zip": img_path_in_zip,
                    "boxes": boxes,
                    "has_defect": len(boxes) > 0
                }

                if len(boxes) > 0:
                    damaged_items.append(item_record)
                else:
                    clean_items.append(item_record)
            except Exception:
                continue

        print(f"[*] Total parsed: {len(damaged_items)} damaged roads, {len(clean_items)} clean roads")

        # Select balanced subset for clean robust generalization
        random.seed(42)
        random.shuffle(damaged_items)
        random.shuffle(clean_items)

        # 1,250 damaged + 1,250 clean for train = 2,500 train
        # 300 damaged + 300 clean for test = 600 test
        train_damaged = damaged_items[:max_train_samples // 2]
        train_clean = clean_items[:max_train_samples // 2]
        test_damaged = damaged_items[max_train_samples // 2 : (max_train_samples // 2) + (max_test_samples // 2)]
        test_clean = clean_items[max_train_samples // 2 : (max_train_samples // 2) + (max_test_samples // 2)]

        train_set = train_damaged + train_clean
        test_set = test_damaged + test_clean
        random.shuffle(train_set)
        random.shuffle(test_set)

        print(f"[*] Target Train: {len(train_set)} images ({len(train_damaged)} damaged, {len(train_clean)} clean)")
        print(f"[*] Target Test:  {len(test_set)} images ({len(test_damaged)} damaged, {len(test_clean)} clean)")

        # Write to destination directories
        def write_subset(items, out_folder):
            for it in items:
                img_data = iz.read(it["img_path_in_zip"])
                img_dest = os.path.join(out_folder, f"{it['id']}.jpg")
                with open(img_dest, "wb") as f:
                    f.write(img_data)

                lbl_dest = os.path.join(out_folder, f"{it['id']}.txt")
                with open(lbl_dest, "w") as f:
                    for b in it["boxes"]:
                        # 0: defect, cx, cy, w, h
                        f.write(f"0 {b[1]:.5f} {b[2]:.5f} {b[3]:.5f} {b[4]:.5f}\n")

        print("[*] Extracting and writing train set...")
        write_subset(train_set, train_out)
        print("[*] Extracting and writing test set...")
        write_subset(test_set, test_out)

    print(f"\n[OK] Indian Road Dataset Ready at {dest_dir}:")
    print(f"  - Train: {len(os.listdir(train_out)) // 2} images + labels")
    print(f"  - Test:  {len(os.listdir(test_out)) // 2} images + labels")

if __name__ == "__main__":
    extract_and_prepare_dataset()
