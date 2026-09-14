import os

# Reduce Ultralytics console output
os.environ["YOLO_VERBOSE"] = "False"

from pathlib import Path
from ultralytics import YOLO
from tqdm import tqdm


# ============================================================
# PATHS / SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_YAML = (
    ROOT
    / "datasets"
    / "merged_dataset"
    / "data.yaml"
)

RUN_NAME = "merged_yolo26n_seg"

CHECKPOINT_DIR = (
    ROOT
    / "checkpoints"
    / RUN_NAME
)

LAST_CHECKPOINT = (
    CHECKPOINT_DIR
    / "weights"
    / "last.pt"
)


# ============================================================
# CUSTOM CLEAN PROGRESS BAR
# ============================================================

progress_bar = None


def on_train_epoch_start(trainer):
    global progress_bar

    if progress_bar is not None:
        progress_bar.close()

    total_batches = len(trainer.train_loader)

    progress_bar = tqdm(
        total=total_batches,
        desc=f"Epoch {trainer.epoch + 1}/{trainer.args.epochs}",
        unit="batch",
        dynamic_ncols=True,
        leave=False,
        position=0,
        mininterval=0.5
    )


def on_train_batch_end(trainer):
    global progress_bar

    if progress_bar is not None:
        progress_bar.update(1)


def on_train_epoch_end(trainer):
    global progress_bar

    if progress_bar is not None:

        progress_bar.n = progress_bar.total
        progress_bar.refresh()
        progress_bar.close()

        progress_bar = None


def on_train_end(trainer):
    global progress_bar

    if progress_bar is not None:
        progress_bar.close()
        progress_bar = None

    print("\nTraining finished successfully.")


def add_callbacks(model):

    model.add_callback(
        "on_train_epoch_start",
        on_train_epoch_start
    )

    model.add_callback(
        "on_train_batch_end",
        on_train_batch_end
    )

    model.add_callback(
        "on_train_epoch_end",
        on_train_epoch_end
    )

    model.add_callback(
        "on_train_end",
        on_train_end
    )


# ============================================================
# TRAINING
# ============================================================

def main():

    print("=" * 70)
    print("MULTI-FOOD SEGMENTATION TRAINING")
    print("=" * 70)

    # --------------------------------------------------------
    # Resume automatically
    # --------------------------------------------------------

    if LAST_CHECKPOINT.exists():

        print(f"\nCheckpoint found:")
        print(LAST_CHECKPOINT)

        print("\nResuming training...\n")

        model = YOLO(
            str(LAST_CHECKPOINT)
        )

        add_callbacks(model)

        model.train(
            resume=True
        )

        return


    # --------------------------------------------------------
    # Fresh training
    # --------------------------------------------------------

    print("\nNo checkpoint found.")
    print("Starting new training...\n")

    model = YOLO(
        "yolo26n-seg.pt"
    )

    add_callbacks(model)

    model.train(

        # Dataset
        data=str(DATA_YAML),

        # Training
        epochs=50,
        imgsz=640,

        # GTX 1660 Super 6 GB
        batch=4,
        device=0,

        # Windows + 16 GB RAM
        workers=0,

        # 1660 Super AMP compatibility issue
        amp=False,

        # Saving
        project=str(
            ROOT / "checkpoints"
        ),

        name=RUN_NAME,

        save=True,

        # Save every epoch
        save_period=1,

        # Stop if no improvement
        patience=12,

        # Generate graphs
        plots=True,

        # Reduce Ultralytics console spam
        verbose=False,

        exist_ok=True
    )


# ============================================================
# START
# ============================================================

if __name__ == "__main__":
    main()