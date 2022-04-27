import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FINETUNE_LEARNING_RATE = 3e-5
LEARNING_RATE = 1e-6
WEIGHT_DECAY = 5e-4
BATCH_SIZE = 4
NUM_PRETRAIN_EPOCHS = 10
NUM_EPOCHS = 15
NUM_SPLITS = 4
NUM_WORKERS = 12
CHECKPOINT_FILE = "b3.pth.tar"
PIN_MEMORY = True
SAVE_MODEL = True
LOAD_MODEL = True

train_transforms = A.Compose(
    [
        A.Resize(width=224, height=224),
        A.transforms.HorizontalFlip(p=0.5),
        A.transforms.GridDistortion(p=0.5),
        A.geometric.rotate.Rotate(limit=15,p=0.5),
        
        ToTensorV2(),
    ],
    additional_targets={'image2': 'image', 'image3': 'image'}
)

val_transforms = A.Compose(
    [
        A.Resize(width=224, height=224),
        ToTensorV2(),
    ],
    additional_targets={'image2': 'image', 'image3': 'image'}
)