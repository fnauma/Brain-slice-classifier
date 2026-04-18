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
CLASS_NAMES = ["Human", "Mouse"]


class BrainClassifier(nn.Module):
	"""Architecture matching the training notebook."""

	def __init__(self, freeze_backbone: bool = False):
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
		return Image.fromarray(arr.astype(np.uint8))
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
	model = BrainClassifier(freeze_backbone=False).to(device)
	missing, unexpected = model.load_state_dict(candidate_state, strict=False)

	print("Loaded state_dict into BrainClassifier")
	if missing:
		print(f"Missing keys ({len(missing)}):", missing[:5], "...")
	if unexpected:
		print(f"Unexpected keys ({len(unexpected)}):", unexpected[:5], "...")

	model.eval()
	return model


def build_transform(img_size: int = 224):
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
	threshold: float = 0.5,
	img_size: int = 224,
):
	tf = build_transform(img_size=img_size)
	img = load_image(image_path)
	x = tf(img).unsqueeze(0).to(device)

	logits = model(x)
	prob_mouse = torch.sigmoid(logits).item()
	pred = int(prob_mouse > threshold)

	return {
		"image_path": image_path,
		"prob_mouse": prob_mouse,
		"threshold": threshold,
		"pred_label": CLASS_NAMES[pred],
		"pred_id": pred,
	}


def parse_args():
	parser = argparse.ArgumentParser(
		description="Run single-slice inference with the trained human-vs-mouse model."
	)
	parser.add_argument("--model", required=True, help="Path to .pt model file")
	parser.add_argument("--image", required=True, help="Path to input slice image")
	parser.add_argument(
		"--device",
		default="auto",
		choices=["auto", "cpu", "cuda"],
		help="Inference device",
	)
	parser.add_argument("--img-size", type=int, default=224, help="Input resize (default: 224)")
	parser.add_argument(
		"--threshold", type=float, default=0.5, help="Decision threshold for mouse class"
	)
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
		threshold=args.threshold,
		img_size=args.img_size,
	)

	print("=== SINGLE SLICE INFERENCE ===")
	print(f"Image      : {result['image_path']}")
	print(f"Device     : {device}")
	print(f"p(mouse)   : {result['prob_mouse']:.6f}")
	print(f"Threshold  : {result['threshold']:.3f}")
	print(f"Prediction : {result['pred_label']} ({result['pred_id']})")


if __name__ == "__main__":
	main()