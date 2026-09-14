from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent

MODEL = ROOT / "checkpoints" / "foodbd_yolo26n_seg" / "weights" / "best.pt"

TEST_IMAGES = ROOT / "datasets" / "FoodBD" / "root" / "FoodBD" / "test" / "images"

OUTPUT = ROOT / "outputs" / "segmentation_test"

model = YOLO(str(MODEL))

results = model.predict(
    source=str(TEST_IMAGES),
    conf=0.25,
    imgsz=640,
    device=0,
    save=True,
    project=str(ROOT / "outputs"),
    name="segmentation_test",
    exist_ok=True
)

print("\nPrediction finished.")
print(f"Results saved to: {OUTPUT}")