from pathlib import Path
import pandas as pd

DATASET = Path("datasets/FoodBD/root/FoodBD")

label_dir = DATASET / "train" / "labels"
image_dir = DATASET / "train" / "images"

labels = list(label_dir.glob("*.txt"))
images = list(image_dir.glob("*.*"))

print("Train images:", len(images))
print("Train labels:", len(labels))

if labels:
    sample = labels[0]

    print("\nSample label file:")
    print(sample)

    print("\nContent:")
    with open(sample, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            print(line.strip())

            if i == 4:
                break


meta_file = Path("datasets/FoodBD/root/FoodBD_Meta_data.csv")

if meta_file.exists():
    df = pd.read_csv(meta_file)

    print("\nMetadata columns:")
    print(df.columns.tolist())

    print("\nFirst rows:")
    print(df.head())