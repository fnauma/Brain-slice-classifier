# %% [markdown]
# Task: classify brain slices as human or mouse.

# %%
# import sys
# !{sys.executable} -m pip install torch torchvision timm tifffile pillow scikit-learn matplotlib seaborn

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
    roc_curve,
    precision_recall_curve,
)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
HUMAN_DIR = "human_brain_slices"
MOUSE_DIR = "mouse_brain_slices"
MODEL_SAVE_PATH = 'model_weights.pt'

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
def show_samples(human_dir, mouse_dir, n=4):
    human_files = [os.path.join(human_dir, f) for f in os.listdir(human_dir)][:n]
    mouse_files  = [os.path.join(mouse_dir,  f) for f in os.listdir(mouse_dir)][:n]

    fig, axes = plt.subplots(2, n, figsize=(4*n, 8))
    for i, path in enumerate(human_files):
        img = load_image(path)
        axes[0, i].imshow(img)
        axes[0, i].set_title(f'HUMAN\n{os.path.basename(path)[:20]}', fontsize=8)
        axes[0, i].axis('off')
    for i, path in enumerate(mouse_files):
        img = load_image(path)
        axes[1, i].imshow(img)
        axes[1, i].set_title(f'MOUSE\n{os.path.basename(path)[:20]}', fontsize=8)
        axes[1, i].axis('off')

    plt.suptitle('Sample images — sanity check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

show_samples(HUMAN_DIR, MOUSE_DIR)

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


def build_file_list(human_dir, mouse_dir):
    paths, labels = [], []
    for f in os.listdir(human_dir):
        paths.append(os.path.join(human_dir, f))
        labels.append(0)   # 0 = human
    for f in os.listdir(mouse_dir):
        paths.append(os.path.join(mouse_dir, f))
        labels.append(1)   # 1 = mouse
    return paths, labels


all_paths, all_labels = build_file_list(HUMAN_DIR, MOUSE_DIR)
print(f'Total images : {len(all_paths)}')
print(f'  Human      : {all_labels.count(0)}')
print(f'  Mouse      : {all_labels.count(1)}')

# %%
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((224, 224)),
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
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

print('Transforms defined.')

# %%
# Deterministic train/val/test split (no leakage from test into training decisions)
TEST_SIZE = 0.20   # final hold-out test
VAL_SIZE  = 0.20   # validation fraction of the remaining train pool
SEED = 42

tr_paths, te_paths, tr_labels, te_labels = train_test_split(
    all_paths,
    all_labels,
    test_size=TEST_SIZE,
    stratify=all_labels,
    shuffle=True,
    random_state=SEED,
 )

tr_paths, va_paths, tr_labels, va_labels = train_test_split(
    tr_paths,
    tr_labels,
    test_size=VAL_SIZE,
    stratify=tr_labels,
    shuffle=True,
    random_state=SEED,
 )

tr_ds = BrainSliceDataset(tr_paths, tr_labels, train_tf)
va_ds = BrainSliceDataset(va_paths, va_labels, val_tf)
te_ds = BrainSliceDataset(te_paths, te_labels, val_tf)

tr_dl = DataLoader(tr_ds, batch_size=8, shuffle=True,  num_workers=0)
va_dl = DataLoader(va_ds, batch_size=8, shuffle=False, num_workers=0)
te_dl = DataLoader(te_ds, batch_size=8, shuffle=False, num_workers=0)

print(f'Train samples : {len(tr_ds)}')
print(f'Val   samples : {len(va_ds)}')
print(f'Test  samples : {len(te_ds)}')
print('Class counts (0=Human, 1=Mouse)')
print(f'  Train: human={tr_labels.count(0)} mouse={tr_labels.count(1)}')
print(f'  Val  : human={va_labels.count(0)} mouse={va_labels.count(1)}')
print(f'  Test : human={te_labels.count(0)} mouse={te_labels.count(1)}')

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
# Final test evaluation with standard classification metrics
model.load_state_dict(torch.load(MODEL_SAVE_PATH, map_location=device))
model.eval()

# TTA transform (with augmentation)
tta_tf = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

def predict_tta(path, n=8):
    img = load_image(path)
    probs = []
    for _ in range(n):
        tensor = tta_tf(img).unsqueeze(0).to(device)
        with torch.no_grad():
            prob = torch.sigmoid(model(tensor)).item()
        probs.append(prob)
    return np.mean(probs)

# Run inference on all test images
all_preds, all_probs = [], []
for path in te_paths:
    prob = predict_tta(path, n=8)
    pred = 1 if prob > 0.5 else 0
    all_preds.append(pred)
    all_probs.append(prob)

y_true = np.array(te_labels)
y_pred = np.array(all_preds)
y_prob = np.array(all_probs)

# Core metrics
acc = (y_true == y_pred).mean()
precision = precision_score(y_true, y_pred, zero_division=0)
recall = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)

# Probability-quality metrics
roc_auc = roc_auc_score(y_true, y_prob)
pr_auc = average_precision_score(y_true, y_prob)
ll = log_loss(y_true, y_prob, labels=[0, 1])
brier = brier_score_loss(y_true, y_prob)

print('=== Overall Metrics ===')
print(f'Accuracy        : {acc:.4f}')
print(f'Precision       : {precision:.4f}')
print(f'Recall          : {recall:.4f}')
print(f'F1-score        : {f1:.4f}')
print(f'ROC-AUC         : {roc_auc:.4f}')
print(f'PR-AUC          : {pr_auc:.4f}')
print(f'Log Loss        : {ll:.4f}')
print(f'Brier Score     : {brier:.4f}')

# Per-class report
print('\n=== Classification Report ===')
print(classification_report(y_true, y_pred, target_names=['Human', 'Mouse'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_true, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=['Human', 'Mouse'], yticklabels=['Human', 'Mouse'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# ROC and Precision-Recall curves
fpr, tpr, _ = roc_curve(y_true, y_prob)
prec_curve, rec_curve, _ = precision_recall_curve(y_true, y_prob)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(fpr, tpr, label=f'ROC-AUC = {roc_auc:.3f}', color='darkorange')
axes[0].plot([0, 1], [0, 1], 'k--', alpha=0.6)
axes[0].set_title('ROC Curve')
axes[0].set_xlabel('False Positive Rate')
axes[0].set_ylabel('True Positive Rate')
axes[0].legend(loc='lower right')

axes[1].plot(rec_curve, prec_curve, label=f'PR-AUC = {pr_auc:.3f}', color='purple')
axes[1].set_title('Precision-Recall Curve')
axes[1].set_xlabel('Recall')
axes[1].set_ylabel('Precision')
axes[1].legend(loc='lower left')

plt.tight_layout()
plt.show()

# Visual audit on predictions
n = len(te_paths)
cols = 6
rows = max(1, (n + cols - 1) // cols)
fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
axes = np.array(axes).reshape(-1)
class_names = ['Human', 'Mouse']

for i, (path, true, pred, prob) in enumerate(zip(te_paths, y_true, y_pred, y_prob)):
    img = load_image(path)
    axes[i].imshow(img)
    confidence = prob if pred == 1 else 1 - prob
    correct = '✓' if pred == true else '✗'
    axes[i].set_title(
        f'{correct} Pred: {class_names[pred]}\nTrue: {class_names[true]} ({confidence:.0%})',
        fontsize=7,
        color='green' if pred == true else 'red'
    )
    axes[i].axis('off')

for j in range(len(te_paths), len(axes)):
    axes[j].axis('off')

plt.suptitle('Randomized Test Split Predictions', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

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

sample_path = te_paths[0] if len(te_paths) > 0 else all_paths[0]

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


