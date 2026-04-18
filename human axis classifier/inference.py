import argparse
import os

import numpy as np
import tifffile
import torch
import torch.nn as nn
import timm
from PIL import Image
from torchvision import transforms


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
CLASS_NAMES = ["coronal", "sagittal", "axial"]
MODEL_NAME = "convnext_tiny"


class AxisClassifier(nn.Module):
	"""Architecture matching the training script."""

	def __init__(self, num_classes: int = len(CLASS_NAMES)):
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


def load_image(path: str) -> Image.Image:
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
		return Image.fromarray(arr.astype(np.uint8)).convert("RGB")
	return Image.open(path).convert("RGB")


def _strip_module_prefix(state_dict):
	if not any(k.startswith("module.") for k in state_dict.keys()):
		return state_dict
	return {k.replace("module.", "", 1): v for k, v in state_dict.items()}


def load_uploaded_model(model_path: str, device: torch.device) -> nn.Module:
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
	model = AxisClassifier(num_classes=len(CLASS_NAMES)).to(device)
	missing, unexpected = model.load_state_dict(candidate_state, strict=False)

	print("Loaded state_dict into AxisClassifier")
	if missing:
		print(f"Missing keys ({len(missing)}):", missing[:5], "...")
	if unexpected:
		print(f"Unexpected keys ({len(unexpected)}):", unexpected[:5], "...")

	model.eval()
	return model


def build_transform(img_size: int = 320):
	return transforms.Compose(
		[
			transforms.Resize((img_size, img_size)),
			transforms.Grayscale(num_output_channels=3),
			transforms.ToTensor(),
			transforms.Normalize(MEAN, STD),
		]
	)


@torch.no_grad()
def predict_slice(
	model: nn.Module,
	image_path: str,
	device: torch.device,
	img_size: int = 320,
):
	tf = build_transform(img_size=img_size)
	img = load_image(image_path)
	x = tf(img).unsqueeze(0).to(device)

	logits = model(x)
	probs = torch.softmax(logits, dim=1).squeeze(0)
	pred_id = int(torch.argmax(probs).item())

	return {
		"image_path": image_path,
		"pred_label": CLASS_NAMES[pred_id],
		"pred_id": pred_id,
		"class_probs": {name: float(probs[i].item()) for i, name in enumerate(CLASS_NAMES)},
	}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run single-slice inference with the trained human-axis classifier."
	)
	parser.add_argument("--model", required=True, help="Path to .pt model file")
	parser.add_argument("--image", required=True, help="Path to input slice image")
	parser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Inference device",
	)
	parser.add_argument("--img-size", type=int, default=320, help="Input resize (default: 320)")
	return parser.parse_args()


def main():
	args = parse_args()

	if args.device == "auto":
		device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	else:
		device = torch.device(args.device)

	model = load_uploaded_model(args.model, device)
	result = predict_slice(
		model=model,
		image_path=args.image,
		device=device,
		img_size=args.img_size,
	)

	print("=== SINGLE SLICE INFERENCE (HUMAN AXIS) ===")
	print(f"Image      : {result['image_path']}")
	print(f"Device     : {device}")
	for cls_name, prob in result["class_probs"].items():
		print(f"p({cls_name:8s}): {prob:.6f}")
	print(f"Prediction : {result['pred_label']} ({result['pred_id']})")


if __name__ == "__main__":
	main()
