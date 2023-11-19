import os 

from pathlib import Path


PROJECT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = PROJECT_DIR / "data"
WEIGHTS_DIR = PROJECT_DIR / "weights"
TRAIN_DIR = DATA_DIR / "train_subset"
VALID_DIR = DATA_DIR / "val_subset"
MEDIA_DIR = PROJECT_DIR / "media"

IMG_TYPE = "jpg"
MASK_TYPE = "png"