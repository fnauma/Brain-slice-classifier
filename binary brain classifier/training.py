# %% [markdown]
#  Task: classify brain slices as human or mouse.

# %%
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import tifffile
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import timm
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    log_loss,
    brier_score_loss,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


# %%
# Kaggle dataset roots (update as required)
HUMAN_ROOT = "/kaggle/input/datasets/fatimanauman/trainhuman/human_brain_slices"
MOUSE_ROOT = "/kaggle/input/datasets/fatimanauman/randomizedmouse/mouse_brain_slices"

AXES = ("coronal", "sagittal", "axial")
HUMAN_DIRS = {axis: os.path.join(HUMAN_ROOT, axis) for axis in AXES}
MOUSE_DIRS = {axis: os.path.join(MOUSE_ROOT, axis) for axis in AXES}

MODEL_SAVE_PATH = 'binary_model_weights.pt'

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
def _collect_files_from_dirs(dirs_map):
    files = []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp')
    for axis, d in dirs_map.items():
        if not os.path.isdir(d):
            print(f'Warning: {axis} dir not found -> {d}')
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in valid_ext:
                files.append(p)
    return files


def show_samples(human_dirs, mouse_dirs, n=4):
    human_files = _collect_files_from_dirs(human_dirs)[:n]
    mouse_files = _collect_files_from_dirs(mouse_dirs)[:n]

    n_show = max(len(human_files), len(mouse_files), 1)
    fig, axes = plt.subplots(2, n_show, figsize=(4 * n_show, 8))
    axes = np.array(axes).reshape(2, n_show)

    for i in range(n_show):
        if i < len(human_files):
            img = load_image(human_files[i])
            axes[0, i].imshow(img)
            axes[0, i].set_title(f'HUMAN\n{os.path.basename(human_files[i])[:20]}', fontsize=8)
        axes[0, i].axis('off')

        if i < len(mouse_files):
            img = load_image(mouse_files[i])
            axes[1, i].imshow(img)
            axes[1, i].set_title(f'MOUSE\n{os.path.basename(mouse_files[i])[:20]}', fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle('Sample images — sanity check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

show_samples(HUMAN_DIRS, MOUSE_DIRS)

# %%
class BrainSliceDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.paths     = image_paths
        self.labels    = labels
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(self.labels[idx], dtype=torch.float32)


def build_file_list(human_dirs, mouse_dirs):
    paths, labels = [], []
    valid_ext = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff', '.jfif', '.webp')

    for axis, d in human_dirs.items():
        if not os.path.isdir(d):
            print(f'Warning: missing human {axis} dir -> {d}')
            continue
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in valid_ext:
                paths.append(p)
                labels.append(0)   # 0 = human

    for axis, d in mouse_dirs.items():
        if not os.path.isdir(d):
            print(f'Warning: missing mouse {axis} dir -> {d}')
            continue
        for f in os.listdir(d):
            p = os.path.join(d, f)
            if os.path.isfile(p) and os.path.splitext(f)[1].lower() in valid_ext:
                paths.append(p)
                labels.append(1)   # 1 = mouse

    return paths, labels


all_paths, all_labels = build_file_list(HUMAN_DIRS, MOUSE_DIRS)
print(f'Total images : {len(all_paths)}')
print(f'  Human      : {all_labels.count(0)}')
print(f'  Mouse      : {all_labels.count(1)}')

# %%
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(180),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print('Transforms defined.')


# %%
# Deterministic train/val split only (testing is done separately below)
VAL_SIZE = 0.20
SEED = 42

tr_paths, va_paths, tr_labels, va_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=VAL_SIZE,
    stratify=all_labels,
    shuffle=True,
    random_state=SEED,
)

tr_ds = BrainSliceDataset(tr_paths, tr_labels, train_tf)
va_ds = BrainSliceDataset(va_paths, va_labels, val_tf)

tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True,  num_workers=0)
va_dl = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0)

print(f'Train samples : {len(tr_ds)}')
print(f'Val   samples : {len(va_ds)}')
print('Class counts (0=Human, 1=Mouse)')
print(f'  Train: human={tr_labels.count(0)} mouse={tr_labels.count(1)}')
print(f'  Val  : human={va_labels.count(0)} mouse={va_labels.count(1)}')

# %%
class BrainClassifier(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        # Pretrained EfficientNet-B0, head removed
        self.backbone = timm.create_model('efficientnet_b0', pretrained=True, num_classes=0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1), # single logit → sigmoid → probability
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

    def unfreeze_top(self, n_blocks=3):
        """Unfreeze last n blocks of backbone for fine-tuning."""
        blocks = list(self.backbone.children())
        for block in blocks[-n_blocks:]:
            for p in block.parameters():
                p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Trainable params after unfreeze: {n_trainable:,}')


model = BrainClassifier(freeze_backbone=True).to(device)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params (head only): {n:,}')


# %%
criterion  = nn.BCEWithLogitsLoss()
optimizer  = torch.optim.Adam(
    filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)
scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5)

EPOCHS          = 40
UNFREEZE_EPOCH  = 5
EARLY_STOP_PAT  = 8

# Early stopping and LR scheduling use validation loss only.
history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_loss   = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    if epoch == UNFREEZE_EPOCH:
        print(f'\n→ Epoch {epoch}: unfreezing backbone top layers...')
        model.unfreeze_top(n_blocks=6)
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=3, factor=0.5)

    # Train
    model.train()
    tr_loss = 0.0
    for imgs, lbls in tr_dl:
        imgs, lbls = imgs.to(device), lbls.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), lbls)
        loss.backward()
        optimizer.step()
        tr_loss += loss.item()

    # Validation split evaluation
    model.eval()
    va_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in va_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out      = model(imgs)
            va_loss += criterion(out, lbls).item()
            preds    = (torch.sigmoid(out) > 0.5).float()
            correct += (preds == lbls).sum().item()
            total   += len(lbls)

    avg_tr = tr_loss / len(tr_dl)
    avg_va = va_loss / len(va_dl)
    acc    = correct / total

    history['train_loss'].append(avg_tr)
    history['val_loss'].append(avg_va)
    history['val_acc'].append(acc)

    scheduler.step(avg_va)
    print(f'Epoch {epoch:02d}/{EPOCHS} | '
          f'train_loss={avg_tr:.4f}  val_loss={avg_va:.4f}  val_acc={acc:.2%}')

    if avg_va < best_val_loss:
        best_val_loss = avg_va
        torch.save(model.state_dict(), MODEL_SAVE_PATH)
        patience_counter = 0
        print(f'  ✓ Saved best model (val_loss={best_val_loss:.4f})')
    else:
        patience_counter += 1
        if patience_counter >= EARLY_STOP_PAT:
            print(f'\nEarly stopping at epoch {epoch}.')
            break

print(f'\nTraining complete. Best val_loss: {best_val_loss:.4f}')


# %%
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# %%
# Latency benchmark: end-to-end (load+transform+infer) and infer-only
import time

model.eval()

def predict_one(path):
    img = load_image(path)
    x = val_tf(img).unsqueeze(0).to(device)
    with torch.no_grad():
        logit = model(x)
        prob = torch.sigmoid(logit).item()
    return prob

sample_path = va_paths[0] if len(va_paths) > 0 else all_paths[0]

# Warmup
for _ in range(20):
    _ = predict_one(sample_path)
if device.type == 'cuda':
    torch.cuda.synchronize()

# End-to-end latency
runs = 200
times_ms = []
for _ in range(runs):
    t0 = time.perf_counter()
    _ = predict_one(sample_path)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    times_ms.append((t1 - t0) * 1000)

times_ms = np.array(times_ms)
print('=== End-to-end single-image latency (load+transform+infer) ===')
print(f'Mean   : {times_ms.mean():.2f} ms')
print(f'Median : {np.median(times_ms):.2f} ms')
print(f'P95    : {np.percentile(times_ms, 95):.2f} ms')
print(f'P99    : {np.percentile(times_ms, 99):.2f} ms')

# Infer-only latency (tensor already prepared)
img = load_image(sample_path)
x = val_tf(img).unsqueeze(0).to(device)

for _ in range(50):
    with torch.no_grad():
        _ = model(x)
if device.type == 'cuda':
    torch.cuda.synchronize()

infer_ms = []
for _ in range(500):
    t0 = time.perf_counter()
    with torch.no_grad():
        _ = model(x)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    t1 = time.perf_counter()
    infer_ms.append((t1 - t0) * 1000)

infer_ms = np.array(infer_ms)
print('\n=== Inference-only latency (tensor ready) ===')
print(f'Mean   : {infer_ms.mean():.2f} ms')
print(f'Median : {np.median(infer_ms):.2f} ms')
print(f'P95    : {np.percentile(infer_ms, 95):.2f} ms')
print(f'P99    : {np.percentile(infer_ms, 99):.2f} ms')

# Model footprint
params = sum(p.numel() for p in model.parameters())
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
model_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024**2)
print('\n=== Model footprint ===')
print(f'Total params      : {params:,}')
print(f'Trainable params  : {trainable:,}')
print(f'Approx weight size: {model_mb:.2f} MB (FP32)')

# %%
# =====================
# CONFIGURATION
# =====================

# Path to uploaded/trained .pt model
MODEL_PATH = "/kaggle/working/binary_model_weights.pt"

# Dataset roots
HUMAN_ROOT = "/kaggle/input/datasets/fatimanauman/standardhuman/standardhuman"
MOUSE_ROOT = "/kaggle/input/datasets/fatimanauman/standardmouse/standardmouse"

AXES = ("axial", "coronal", "sagittal")
HUMAN_DIRS = {axis: os.path.join(HUMAN_ROOT, axis) for axis in AXES}
MOUSE_DIRS = {axis: os.path.join(MOUSE_ROOT, axis) for axis in AXES}

# Inference settings
BATCH_SIZE = 16
NUM_WORKERS = 0
IMG_SIZE = 224
THRESHOLD = 0.5

print("Model path:", MODEL_PATH)
print("Human dirs:", HUMAN_DIRS)
print("Mouse dirs:", MOUSE_DIRS)

for axis in AXES:
    if not os.path.isdir(HUMAN_DIRS[axis]):
        print(f"Warning: Missing HUMAN {axis} directory -> {HUMAN_DIRS[axis]}")
    if not os.path.isdir(MOUSE_DIRS[axis]):
        print(f"Warning: Missing MOUSE {axis} directory -> {MOUSE_DIRS[axis]}")

# %%
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".jfif", ".webp")

def load_image(path):
    """Load JPG/PNG/TIFF/etc. as RGB PIL image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in (".tif", ".tiff"):
        arr = tifffile.imread(path)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8))
    return Image.open(path).convert("RGB")

def build_file_list(human_dirs, mouse_dirs):
    paths, labels, axes_used = [], [], []

    for axis, d in human_dirs.items():
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in VALID_EXT:
                paths.append(p)
                labels.append(0)  # 0 = human
                axes_used.append(axis)

    for axis, d in mouse_dirs.items():
        if not os.path.isdir(d):
            continue
        for name in os.listdir(d):
            p = os.path.join(d, name)
            if os.path.isfile(p) and os.path.splitext(name)[1].lower() in VALID_EXT:
                paths.append(p)
                labels.append(1)  # 1 = mouse
                axes_used.append(axis)

    return paths, labels, axes_used

class BrainSliceDataset(Dataset):
    def __init__(self, image_paths, labels, axes_used, transform=None):
        self.paths = image_paths
        self.labels = labels
        self.axes_used = axes_used
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = load_image(self.paths[idx])
        if self.transform:
            img = self.transform(img)
        return (
            img,
            torch.tensor(self.labels[idx], dtype=torch.float32),
            self.paths[idx],
            self.axes_used[idx],
        )

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

test_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

all_paths, all_labels, all_axes = build_file_list(HUMAN_DIRS, MOUSE_DIRS)
print(f"Total images: {len(all_paths)}")
print(f"Human count: {sum(1 for x in all_labels if x == 0)}")
print(f"Mouse count: {sum(1 for x in all_labels if x == 1)}")

test_ds = BrainSliceDataset(all_paths, all_labels, all_axes, transform=test_tf)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
print(f"Test samples: {len(test_ds)}")

# %%
class BrainClassifier(nn.Module):
    """Architecture used in training notebook."""
    def __init__(self, freeze_backbone=False):
        super().__init__()
        self.backbone = timm.create_model("efficientnet_b0", pretrained=False, num_classes=0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(self.backbone.num_features, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats).squeeze(1)

def _strip_module_prefix(state_dict):
    if not any(k.startswith("module.") for k in state_dict.keys()):
        return state_dict
    return {k.replace("module.", "", 1): v for k, v in state_dict.items()}

def load_uploaded_model(model_path, device):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    ckpt = torch.load(model_path, map_location=device)

    # Case 1: full serialized model
    if isinstance(ckpt, nn.Module):
        model = ckpt.to(device)
        model.eval()
        print("Loaded full model object from .pt")
        return model

    # Case 2: checkpoint/state_dict
    candidate_state = None
    if isinstance(ckpt, dict):
        for k in ["state_dict", "model_state_dict", "model"]:
            if k in ckpt and isinstance(ckpt[k], dict):
                candidate_state = ckpt[k]
                break
        if candidate_state is None and all(torch.is_tensor(v) for v in ckpt.values()):
            candidate_state = ckpt

    if candidate_state is None:
        raise ValueError(
            "Could not detect model weights in .pt. Expected a full model or a state_dict checkpoint."
        )

    candidate_state = _strip_module_prefix(candidate_state)
    model = BrainClassifier(freeze_backbone=False).to(device)
    missing, unexpected = model.load_state_dict(candidate_state, strict=False)

    print("Loaded state_dict into BrainClassifier")
    if missing:
        print(f"Missing keys ({len(missing)}):", missing[:5], "...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}):", unexpected[:5], "...")

    model.eval()
    return model

# %%
# =====================
# TESTING / INFERENCE
# =====================
if len(test_ds) == 0:
    raise RuntimeError("No test images found. Check your folder paths and file extensions.")
model = load_uploaded_model(MODEL_PATH, device)
all_true = []
all_pred = []
all_prob = []
all_path = []
all_axis = []
with torch.no_grad():
    for imgs, labels, paths, axes_used in test_dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > THRESHOLD).astype(int)
        all_true.extend(labels.numpy().astype(int).tolist())
        all_pred.extend(preds.tolist())
        all_prob.extend(probs.tolist())
        all_path.extend(list(paths))
        all_axis.extend(list(axes_used))
y_true = np.array(all_true)
y_pred = np.array(all_pred)
y_prob = np.array(all_prob)
# Core metrics
acc = (y_true == y_pred).mean()
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
# Probability-quality metrics (safe for edge cases)
try:
    roc_auc = roc_auc_score(y_true, y_prob)
except Exception:
    roc_auc = float("nan")
try:
    pr_auc = average_precision_score(y_true, y_prob)
except Exception:
    pr_auc = float("nan")
try:
    ll = log_loss(y_true, y_prob, labels=[0, 1])
except Exception:
    ll = float("nan")
try:
    brier = brier_score_loss(y_true, y_prob)
except Exception:
    brier = float("nan")
print("=== TEST SUMMARY ===")
print(f"Total tested images : {len(y_true)}")
print(f"Human images        : {(y_true == 0).sum()}")
print(f"Mouse images        : {(y_true == 1).sum()}")
print(f"Accuracy            : {acc:.4f}")
print(f"Precision           : {precision:.4f}")
print(f"Recall              : {recall:.4f}")
print(f"F1-score            : {f1:.4f}")
print(f"ROC-AUC             : {roc_auc:.4f}")
print(f"PR-AUC              : {pr_auc:.4f}")
print(f"Log Loss            : {ll:.4f}")
print(f"Brier Score         : {brier:.4f}")
print("\n=== Classification Report ===")
print(classification_report(y_true, y_pred, target_names=["Human", "Mouse"], digits=4))
cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
plt.figure(figsize=(5, 4))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    cbar=False,
    xticklabels=["Human", "Mouse"],
    yticklabels=["Human", "Mouse"],
)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
print("\n=== Axis-wise accuracy ===")
for axis in AXES:
    idx = [i for i, a in enumerate(all_axis) if a == axis]
    if len(idx) == 0:
        print(f"{axis:<8}: no images")
        continue
    axis_acc = (y_true[idx] == y_pred[idx]).mean()
    print(f"{axis:<8}: {axis_acc:.4f} ({len(idx)} images)")

print("\n=== Sample mistakes (up to 10) ===")
mistake_idx = np.where(y_true != y_pred)[0][:10]
class_names = ["Human", "Mouse"]
if len(mistake_idx) == 0:
    print("No misclassifications.")
else:
    for i in mistake_idx:
        print(
            f"{os.path.basename(all_path[i])} | axis={all_axis[i]} | true={class_names[y_true[i]]} | pred={class_names[y_pred[i]]} | p(mouse)={y_prob[i]:.3f}"
        )

    # ── Visual display of misclassified images ──────────────────────────────
    n_mistakes = len(mistake_idx)
    n_cols = min(5, n_mistakes)
    n_rows = (n_mistakes + n_cols - 1) // n_cols          # ceil division

    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(4 * n_cols, 4 * n_rows))
    axes = np.array(axes).reshape(-1)                      # always 1-D

    for plot_i, sample_i in enumerate(mistake_idx):
        img_path = all_path[sample_i]
        true_label = class_names[y_true[sample_i]]
        pred_label = class_names[y_pred[sample_i]]
        prob_mouse = y_prob[sample_i]
        axis_tag = all_axis[sample_i]

        # Load image — works for .png / .jpg / .nii.gz slices stored as images
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            axes[plot_i].set_visible(False)
            continue

        axes[plot_i].imshow(img, cmap="gray")
        axes[plot_i].set_title(
            f"True: {true_label}\nPred: {pred_label}  p={prob_mouse:.2f}\n"
            f"axis={axis_tag}  {os.path.basename(img_path)}",
            fontsize=8,
            color="red",
        )
        axes[plot_i].axis("off")

    # Hide any unused subplot slots
    for j in range(n_mistakes, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle("Misclassified Images", fontsize=13, fontweight="bold", color="red")
    plt.tight_layout()
    plt.show()

# %%
# =====================
# GENERAL RESULTS GRID
# =====================
print("\n=== General results grid ===")
class_names = ["Human", "Mouse"]

# Require prediction outputs from the testing/inference cell
required_vars = ["y_true", "y_pred", "y_prob"]
missing = [v for v in required_vars if v not in globals()]
if missing:
    print("Missing variables:", ", ".join(missing))
    print("Run the TESTING / INFERENCE cell first, then run this cell.")
else:
    # Backward-compatible names
    img_paths = all_path if "all_path" in globals() else all_paths if "all_paths" in globals() else None
    axes_tags = all_axis if "all_axis" in globals() else all_axes if "all_axes" in globals() else None

    if img_paths is None or axes_tags is None:
        print("Missing image path/axis lists. Run the TESTING / INFERENCE cell first.")
    else:
        y_true_arr = np.asarray(y_true)
        y_pred_arr = np.asarray(y_pred)
        y_prob_arr = np.asarray(y_prob)

        # Keep lengths aligned to avoid indexing errors
        n_total = min(len(y_true_arr), len(y_pred_arr), len(y_prob_arr), len(img_paths), len(axes_tags))
        if n_total == 0:
            print("No test predictions available.")
        else:
            n_show = min(20, n_total)
            sample_idx = np.linspace(0, n_total - 1, n_show, dtype=int)

            n_cols = min(5, n_show)
            n_rows = (n_show + n_cols - 1) // n_cols

            fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 4 * n_rows))
            axes = np.array(axes).reshape(-1)

            for plot_i, sample_i in enumerate(sample_idx):
                img_path = img_paths[sample_i]
                true_label = class_names[int(y_true_arr[sample_i])]
                pred_label = class_names[int(y_pred_arr[sample_i])]
                prob_mouse = float(y_prob_arr[sample_i])
                axis_tag = axes_tags[sample_i]
                is_correct = (int(y_true_arr[sample_i]) == int(y_pred_arr[sample_i]))

                try:
                    img = Image.open(img_path).convert("RGB")
                    axes[plot_i].imshow(img)
                    axes[plot_i].set_title(
                        f"T:{true_label} | P:{pred_label} | p={prob_mouse:.2f}\n"
                        f"axis={axis_tag} | {os.path.basename(img_path)}",
                        fontsize=8,
                        color=("green" if is_correct else "red"),
                    )
                    axes[plot_i].axis("off")
                except Exception:
                    axes[plot_i].set_visible(False)

            for j in range(n_show, len(axes)):
                axes[j].set_visible(False)

            fig.suptitle("General Prediction Results", fontsize=13, fontweight="bold")
            plt.tight_layout()
            plt.show()


