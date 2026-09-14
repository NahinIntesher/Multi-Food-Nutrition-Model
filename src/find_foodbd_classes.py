# Which Food Id = Which Food Name
from pathlib import Path
import pandas as pd

ROOT = Path("datasets/FoodBD/root")

print("Possible class/config files:\n")

extensions = ["*.yaml", "*.yml", "*.json", "*.names"]

for ext in extensions:
    for file in ROOT.rglob(ext):
        print(file)

meta_path = ROOT / "FoodBD_Meta_data.csv"

df = pd.read_csv(meta_path)

foods = set()

for value in df["instances"].dropna():
    for food in str(value).split(","):
        foods.add(food.strip())

print("\nUnique food names found:", len(foods))

for i, food in enumerate(sorted(foods)):
    print(i, food)