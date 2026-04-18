# %%
import os
from collections import Counter
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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# %%
# Directories — one per axis
CORONAL_DIR   = "/kaggle/input/datasets/fatimanauman/trainhuman/human_brain_slices/coronal"
SAGITTAL_DIR  = "/kaggle/input/datasets/fatimanauman/trainhuman/human_brain_slices/sagittal"
AXIAL_DIR     = "/kaggle/input/datasets/fatimanauman/trainhuman/human_brain_slices/axial"

MODEL_SAVE_PATH = 'human_axis_classifier.pt'

# Label mapping
CLASSES     = ['coronal', 'sagittal', 'axial']
NUM_CLASSES = len(CLASSES)

# Model / image settings
MODEL_NAME = 'convnext_tiny'
IMG_SIZE   = 320

# Training hyperparams
BATCH_SIZE      = 8
EPOCHS          = 40
UNFREEZE_EPOCH  = 5
EARLY_STOP_PAT  = 8
VAL_SIZE        = 0.20
SEED            = 42

print(f'Classes: {CLASSES}')
print(f'Num classes: {NUM_CLASSES}')
print(f'Model: {MODEL_NAME} | Image size: {IMG_SIZE}x{IMG_SIZE}')

# %%
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb limit

def load_image(path):
    """Load any image format → RGB PIL Image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
        if arr.ndim == 2:                          # grayscale → RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW → HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:                     # drop alpha
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8))
    else:
        return Image.open(path).convert('RGB')

# %%
def show_samples(dirs, class_names, n=4):
    fig, axes = plt.subplots(len(class_names), n, figsize=(4*n, 4*len(class_names)))
    for row, (d, name) in enumerate(zip(dirs, class_names)):
        files = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')][:n]
        for col, path in enumerate(files):
            img = load_image(path)
            axes[row, col].imshow(img)
            axes[row, col].set_title(f'{name.upper()}\n{os.path.basename(path)[:20]}', fontsize=8)
            axes[row, col].axis('off')
    plt.suptitle('Sample images — sanity check', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.show()

show_samples(
    [CORONAL_DIR, SAGITTAL_DIR, AXIAL_DIR],
    CLASSES
)

# %%
def build_file_list(dirs, class_names):
    paths, labels = [], []
    for label, (d, name) in enumerate(zip(dirs, class_names)):
        files = [os.path.join(d, f) for f in os.listdir(d) if not f.startswith('.')]
        paths.extend(files)
        labels.extend([label] * len(files))
        print(f'  {name:12s}: {len(files)} images  (label={label})')
    return paths, labels

print('Building file list...')
all_paths, all_labels = build_file_list(
    [CORONAL_DIR, SAGITTAL_DIR, AXIAL_DIR],
    CLASSES
)
print(f'\nTotal images: {len(all_paths)}')

# %%
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomAffine(
        degrees=10,
        translate=(0.03, 0.03),
        scale=(0.95, 1.05),
        shear=5,
    ),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

val_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

# Train/validation split only (testing is done separately below)
tr_paths, va_paths, tr_labels, va_labels = train_test_split(
    all_paths, all_labels,
    test_size=VAL_SIZE, stratify=all_labels,
    shuffle=True, random_state=SEED
)

tr_counts = np.bincount(tr_labels, minlength=NUM_CLASSES)
print(f'Class counts (train): {dict(zip(CLASSES, tr_counts.tolist()))}')

print(f'Train : {len(tr_paths)}')
print(f'Val   : {len(va_paths)}')
for i, name in enumerate(CLASSES):
    print(f'  {name:12s} — train={tr_labels.count(i)}  val={va_labels.count(i)}')

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
        return img, torch.tensor(self.labels[idx], dtype=torch.long)  # long for CrossEntropyLoss


tr_ds = BrainSliceDataset(tr_paths, tr_labels, train_tf)
va_ds = BrainSliceDataset(va_paths, va_labels, val_tf)

tr_dl = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0)
va_dl = DataLoader(va_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f'Train batches : {len(tr_dl)}')
print(f'Val   batches : {len(va_dl)}')

# %%
class AxisClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, freeze_backbone=True):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=True, num_classes=0)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

    def unfreeze_backbone(self, pct=0.5):
        params = list(self.backbone.parameters())
        start = int((1.0 - pct) * len(params))
        for p in params[start:]:
            p.requires_grad = True
        n_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f'  Trainable params after unfreeze: {n_trainable:,}')


model = AxisClassifier(freeze_backbone=True).to(device)
n = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f'Trainable params (head only): {n:,}')

# %%
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()), lr=5e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=3, factor=0.5)

history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
best_val_loss    = float('inf')
patience_counter = 0

for epoch in range(1, EPOCHS + 1):

    if epoch == UNFREEZE_EPOCH:
        print(f'\n→ Epoch {epoch}: unfreezing backbone top layers...')
        model.unfreeze_backbone(pct=0.5)
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, model.parameters()), lr=1e-5, weight_decay=1e-4)
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
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        tr_loss += loss.item()

    # Validation
    model.eval()
    va_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, lbls in va_dl:
            imgs, lbls = imgs.to(device), lbls.to(device)
            out      = model(imgs)
            va_loss += criterion(out, lbls).item()
            preds    = out.argmax(dim=1)
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
epochs_ran = len(history['train_loss'])
x = range(1, epochs_ran + 1)

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(x, history['train_loss'], label='Train Loss', color='steelblue')
axes[0].plot(x, history['val_loss'],   label='Val Loss',   color='darkorange')
axes[0].axvline(x=UNFREEZE_EPOCH, color='gray', linestyle='--', alpha=0.7, label=f'Unfreeze (epoch {UNFREEZE_EPOCH})')
axes[0].set_title('Loss Curves')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(x, history['val_acc'], label='Val Accuracy', color='mediumseagreen')
axes[1].axvline(x=UNFREEZE_EPOCH, color='gray', linestyle='--', alpha=0.7, label=f'Unfreeze (epoch {UNFREEZE_EPOCH})')
axes[1].set_title('Validation Accuracy')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.suptitle('Mouse Axis Classifier — Training History', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.show()

# %%
CORONAL_DIR  = "/kaggle/input/datasets/fatimanauman/standardhuman/standardhuman/coronal"
SAGITTAL_DIR = "/kaggle/input/datasets/fatimanauman/standardhuman/standardhuman/sagittal"
AXIAL_DIR    = "/kaggle/input/datasets/fatimanauman/standardhuman/standardhuman/axial"

# Path to uploaded/trained .pt file
MODEL_PATH = "/kaggle/working/human_axis_classifier.pt"

# Class order must match training
CLASSES = ['coronal', 'sagittal', 'axial']
NUM_CLASSES = len(CLASSES)

MODEL_NAME = 'convnext_tiny'
IMG_SIZE = 320
BATCH_SIZE = 16

# Folder name keys are treated as ground-truth labels for testing stats
FOLDER_MAP = {
    'coronal': CORONAL_DIR,
    'sagittal': SAGITTAL_DIR,
    'axial': AXIAL_DIR,
}

for folder_label, folder_path in FOLDER_MAP.items():
    print(f"{folder_label}: {folder_path} | exists={os.path.isdir(folder_path)}")

print(f"Model file: {MODEL_PATH} | exists={os.path.isfile(MODEL_PATH)}")

# %%
Image.MAX_IMAGE_PIXELS = None  # disable decompression bomb limit
VALID_EXTS = ('.tif', '.tiff', '.png', '.jpg', '.jpeg', '.jfif', '.bmp')

def is_image_file(path):
    return os.path.splitext(path)[1].lower() in VALID_EXTS

def load_image(path):
    """Load any supported image format -> RGB PIL image."""
    ext = os.path.splitext(path)[1].lower()
    if ext in ('.tif', '.tiff'):
        arr = tifffile.imread(path)
        if arr.ndim == 2:  # grayscale -> RGB
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[0] in (1, 3, 4):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.shape[-1] == 4:  # drop alpha
            arr = arr[..., :3]
        arr = arr.astype(np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8) * 255
        return Image.fromarray(arr.astype(np.uint8)).convert('RGB')
    return Image.open(path).convert('RGB')

# %%
MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]

infer_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(MEAN, STD),
])

class AxisClassifier(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = timm.create_model(MODEL_NAME, pretrained=False, num_classes=0)
        self.head = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(self.backbone.num_features, 128),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        feats = self.backbone(x)
        return self.head(feats)

def load_model(model_path):
    if not os.path.isfile(model_path):
        raise FileNotFoundError(f'Model file not found: {model_path}')

    model = AxisClassifier(num_classes=NUM_CLASSES).to(device)
    ckpt = torch.load(model_path, map_location=device)

    # Supports plain state_dict or wrapped checkpoint
    state_dict = ckpt.get('state_dict', ckpt) if isinstance(ckpt, dict) else ckpt
    model.load_state_dict(state_dict)
    model.eval()
    return model

model = load_model(MODEL_PATH)
print(f'Loaded model from: {MODEL_PATH}')

# %%
def gather_labeled_images(folder_map):
    records = []
    for true_label, folder_path in folder_map.items():
        if true_label not in CLASSES:
            raise ValueError(f"Folder key '{true_label}' is not in CLASSES={CLASSES}")

        if not os.path.isdir(folder_path):
            print(f'[WARN] Missing folder: {folder_path}')
            continue

        files = sorted(os.listdir(folder_path))
        img_files = [f for f in files if not f.startswith('.') and is_image_file(f)]

        for fname in img_files:
            records.append({
                'path': os.path.join(folder_path, fname),
                'source_folder': true_label,
                'true_class': true_label,
                'true_class_idx': CLASSES.index(true_label),
                'filename': fname,
            })
        print(f'{true_label:10s} -> {len(img_files)} images')
    return records

records = gather_labeled_images(FOLDER_MAP)
print(f'\nTotal images found: {len(records)}')
if len(records) == 0:
    raise RuntimeError('No valid images found in the provided folders.')

class LabeledSliceDataset(Dataset):
    def __init__(self, records, transform):
        self.records = records
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        img = load_image(rec['path'])
        img = self.transform(img)
        return img, rec['path'], rec['source_folder'], rec['filename'], rec['true_class'], rec['true_class_idx']

ds = LabeledSliceDataset(records, infer_tf)
dl = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
print(f'Inference batches: {len(dl)}')

# %%
pred_records = []

with torch.no_grad():
    for imgs, paths, source_folders, filenames, true_classes, true_class_idx in dl:
        imgs = imgs.to(device)
        logits = model(imgs)
        probs = torch.softmax(logits, dim=1)

        confs, pred_idx = probs.max(dim=1)

        for i in range(len(paths)):
            pidx = int(pred_idx[i].item())
            tcls = true_classes[i]
            pred_records.append({
                'path': paths[i],
                'filename': filenames[i],
                'source_folder': source_folders[i],
                'true_class': tcls,
                'true_class_idx': int(true_class_idx[i].item()),
                'pred_class_idx': pidx,
                'pred_class': CLASSES[pidx],
                'confidence': float(confs[i].item()),
                'is_correct': (CLASSES[pidx] == tcls),
            })

print(f'Inference complete for {len(pred_records)} images.')

# ===== Stats: correctness =====
n_total = len(pred_records)
n_correct = sum(int(r['is_correct']) for r in pred_records)
n_wrong = n_total - n_correct
acc = n_correct / n_total if n_total else 0.0

print('\n=== Test Accuracy ===')
print(f'Correct   : {n_correct}')
print(f'Incorrect : {n_wrong}')
print(f'Accuracy  : {acc:.4f} ({acc:.2%})')

# ===== Stats: overall predicted class distribution =====
overall_counts = Counter(r['pred_class'] for r in pred_records)
print('\n=== Overall Predicted Class Distribution ===')
for c in CLASSES:
    n = overall_counts.get(c, 0)
    pct = (n / n_total) * 100 if n_total else 0.0
    print(f'{c:10s}: {n:5d} ({pct:6.2f}%)')

# ===== Stats: per true class (recall) =====
print('\n=== Per-Class Recall (by true folder label) ===')
for c in CLASSES:
    recs = [r for r in pred_records if r['true_class'] == c]
    denom = len(recs)
    num = sum(int(r['is_correct']) for r in recs)
    recall = num / denom if denom else 0.0
    print(f'{c:10s}: {num:5d}/{denom:5d} = {recall:.4f} ({recall:.2%})')

# ===== Stats: per predicted class (precision) =====
print('\n=== Per-Class Precision (by predicted class) ===')
for c in CLASSES:
    recs = [r for r in pred_records if r['pred_class'] == c]
    denom = len(recs)
    num = sum(1 for r in recs if r['true_class'] == c)
    precision = num / denom if denom else 0.0
    print(f'{c:10s}: {num:5d}/{denom:5d} = {precision:.4f} ({precision:.2%})')

# ===== Stats: confusion matrix =====
cm = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=int)  # rows=true, cols=pred
for r in pred_records:
    cm[r['true_class_idx'], r['pred_class_idx']] += 1

print('\n=== Confusion Matrix (rows=true, cols=pred) ===')
print('          ' + '  '.join([f'{c[:8]:>8s}' for c in CLASSES]))
for i, c in enumerate(CLASSES):
    row = '  '.join([f'{v:8d}' for v in cm[i]])
    print(f'{c[:8]:>8s}  {row}')

# ===== Stats: confidence =====
conf = np.array([r['confidence'] for r in pred_records], dtype=np.float32)
print('\n=== Confidence Stats (max softmax probability) ===')
print(f'Mean   : {conf.mean():.4f}')
print(f'Median : {np.median(conf):.4f}')
print(f'Std    : {conf.std():.4f}')
print(f'Min    : {conf.min():.4f}')
print(f'Max    : {conf.max():.4f}')

print('\n=== Confidence by Predicted Class ===')
for c in CLASSES:
    vals = [r['confidence'] for r in pred_records if r['pred_class'] == c]
    if len(vals) == 0:
        print(f'{c:10s}: no predictions')
        continue
    vals = np.array(vals, dtype=np.float32)
    print(f'{c:10s}: mean={vals.mean():.4f}  std={vals.std():.4f}  min={vals.min():.4f}  max={vals.max():.4f}')

# Keep arrays handy for any extra analysis
y_true = np.array([r['true_class_idx'] for r in pred_records], dtype=np.int64)
y_pred = np.array([r['pred_class_idx'] for r in pred_records], dtype=np.int64)

# %%
# Optional visual summaries
plt.figure(figsize=(7, 4))
labels = CLASSES
counts = [overall_counts.get(c, 0) for c in labels]
sns.barplot(x=labels, y=counts, palette='viridis')
plt.title('Overall Predicted Class Counts')
plt.xlabel('Predicted Class')
plt.ylabel('Count')
plt.tight_layout()
plt.show()

# Confusion matrix heatmap (rows=true, cols=pred)
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
            xticklabels=CLASSES, yticklabels=CLASSES)
plt.title('Confusion Matrix (Testing)')
plt.xlabel('Predicted')
plt.ylabel('True')
plt.tight_layout()
plt.show()

# Show wrong predictions first (lowest confidence among wrong), then uncertain samples
wrong = [r for r in pred_records if not r['is_correct']]
if len(wrong) > 0:
    k_wrong = min(12, len(wrong))
    wrong_sorted = sorted(wrong, key=lambda x: x['confidence'])[:k_wrong]

    cols = 4
    rows = (k_wrong + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(wrong_sorted):
        img = load_image(rec['path'])
        axes[i].imshow(img)
        axes[i].set_title(
            f"True: {rec['true_class']}\nPred: {rec['pred_class']} ({rec['confidence']:.2%})",
            fontsize=8,
            color='red'
        )
        axes[i].axis('off')

    for j in range(k_wrong, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Wrong Predictions (Lowest Confidence)', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()
else:
    print('No wrong predictions found.')

# Lowest-confidence samples overall
k = min(12, len(pred_records))
worst = sorted(pred_records, key=lambda x: x['confidence'])[:k]

if k > 0:
    cols = 4
    rows = (k + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3.5 * rows))
    axes = np.array(axes).reshape(-1)

    for i, rec in enumerate(worst):
        img = load_image(rec['path'])
        axes[i].imshow(img)
        color = 'green' if rec['is_correct'] else 'red'
        axes[i].set_title(
            f"True: {rec['true_class']}\nPred: {rec['pred_class']} ({rec['confidence']:.2%})",
            fontsize=8,
            color=color
        )
        axes[i].axis('off')

    for j in range(k, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Lowest-Confidence Predictions', fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.show()


