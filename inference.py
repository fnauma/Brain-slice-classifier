# %% [markdown]
# Task: classify brain slices as human or mouse.

# %%
# import sys
# !{sys.executable} -m pip install torch torchvision timm tifffile pillow scikit-learn matplotlib seaborn

# %%
import os
import numpy as np
from PIL import Image
import tifffile
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torchvision import transforms
import timm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
MODEL_SAVE_PATH = 'model_weights.pt'
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
CLASS_NAMES = ['Human', 'Mouse']

# %%
from PIL import Image
Image.MAX_IMAGE_PIXELS = None # disable the decompression bomb limit

def load_image(path):
    """Load any image format → RGB PIL Image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
        if arr.ndim == 2: # grayscale → RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4): # CHW → HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4: # drop alpha
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8))
    else:
        return Image.open(path).convert('RGB')

# %%
val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print('Transforms defined.')

# %%
class BrainClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

# %%
model = BrainClassifier().to(device)
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

print('Model loaded for inference.')

# %%
tta_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def predict_one(path):
    img = load_image(path)
    x = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        prob = torch.sigmoid(model(x)).item()
    pred = 1 if prob > 0.5 else 0
    return CLASS_NAMES[pred], prob

def predict_tta(path, n=8):
    img = load_image(path)
    probs = []
    for _ in range(n):
        x = tta_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            probs.append(torch.sigmoid(model(x)).item())
    prob = float(np.mean(probs))
    pred = 1 if prob > 0.5 else 0
    return CLASS_NAMES[pred], prob

# %%
test_path_1 = "human_brain_slices/images (1).jfif"
test_path_2 = "mouse_brain_slices/images (1).jfif"

print("Single inference:")
print(test_path_1, "->", predict_one(test_path_1))
print(test_path_2, "->", predict_one(test_path_2))

print("\nTTA inference:")
print(test_path_1, "->", predict_tta(test_path_1, n=8))
print(test_path_2, "->", predict_tta(test_path_2, n=8))


