# ---------------- SETUP ----------------
import os
import random
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, TensorDataset
import torchvision.transforms as T
from tqdm import tqdm
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
ROOT = "/Users/jhalakgupta/Downloads/Off-Road/dataset"
IMG_SIZE = 224
PATCH = 14
BATCH_SIZE = 16
EPOCHS = 5
LR = 1e-3
NUM_CLASSES = 10
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SAVE_DIR = "RESULTS"
CACHE_DIR = "CACHE"
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(CACHE_DIR, exist_ok=True)

torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# ---------------- LABEL MAP ----------------
CLASS_MAP = {
    100: 0, 200: 1, 300: 2, 500: 3, 550: 4,
    600: 5, 700: 6, 800: 7, 7100: 8, 10000: 9
}

COLORS = np.array([
    [0, 0, 0],        # class 0
    [128, 64, 128],   # class 1
    [244, 35, 232],   # class 2
    [70, 70, 70],     # class 3
    [102, 102, 156],  # class 4
    [190, 153, 153],  # class 5
    [153, 153, 153],  # class 6
    [250, 170, 30],   # class 7
    [220, 220, 0],    # class 8
    [107, 142, 35],   # class 9
], dtype=np.uint8)

CLASS_NAMES = [
    "Trees", "Lush Bushes", "Dry Grass", "Dry Bushes", "Ground Clutter",
    "Flowers", "Logs", "Rocks", "Landscape", "Sky"
]

# ---------------- DATASET ----------------
class SegDataset(Dataset):
    def __init__(self, root, split):
        self.img_dir = os.path.join(root, split, "Color_Images")
        self.mask_dir = os.path.join(root, split, "Segmentation")
        self.names = sorted(os.listdir(self.img_dir))

        self.img_tf = T.Compose([
            T.Resize((IMG_SIZE, IMG_SIZE)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406],
                        [0.229, 0.224, 0.225])
        ])

        self.mask_tf = T.Resize(
            (IMG_SIZE, IMG_SIZE),
            interpolation=T.InterpolationMode.NEAREST
        )

    def __len__(self):
        return len(self.names)

    def encode_mask(self, mask):
        arr = np.array(mask, dtype=np.int64)
        out = np.zeros_like(arr)
        for k, v in CLASS_MAP.items():
            out[arr == k] = v
        return torch.from_numpy(out)

    def __getitem__(self, idx):
        name = self.names[idx]
        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        mask = Image.open(os.path.join(self.mask_dir, name))

        return (
            self.img_tf(img),
            self.encode_mask(self.mask_tf(mask))
        )

# ---------------- MODEL ----------------
class LinearSegHead(nn.Module):
    def __init__(self, dim, num_classes, H, W):
        super().__init__()
        self.H, self.W = H, W
        self.head = nn.Sequential(
            nn.Conv2d(dim, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, num_classes, 1)
        )

    def forward(self, x):
        B, N, C = x.shape
        x = x.view(B, self.H, self.W, C).permute(0, 3, 1, 2)
        return self.head(x)

# ---------------- METRICS ----------------
@torch.no_grad()
def compute_iou(pred, target, num_classes):
    ious = []
    for c in range(num_classes):
        inter = ((pred == c) & (target == c)).sum()
        union = ((pred == c) | (target == c)).sum()
        if union > 0:
            ious.append((inter / union).item())
    return np.mean(ious) if ious else 0.0

def compute_class_weights(masks):
    flat = torch.cat([m.flatten() for m in masks])
    counts = torch.bincount(flat, minlength=NUM_CLASSES).float()
    weights = torch.sqrt(counts.sum() / (counts + 1e-6))
    return (weights / weights.mean()).to(DEVICE)

# ---------------- FEATURE CACHING ----------------
@torch.no_grad()
def cache_features(backbone, loader, split):
    feat_path = f"{CACHE_DIR}/{split}_feats.pt"
    mask_path = f"{CACHE_DIR}/{split}_masks.pt"

    if os.path.exists(feat_path):
        return torch.load(feat_path), torch.load(mask_path)

    feats, masks = [], []
    for imgs, lbls in tqdm(loader, desc=f"Caching {split}"):
        imgs = imgs.to(DEVICE)
        tokens = backbone.forward_features(imgs)["x_norm_patchtokens"]
        feats.append(tokens.cpu())
        masks.append(lbls)

    feats = torch.cat(feats)
    masks = torch.cat(masks)

    torch.save(feats, feat_path)
    torch.save(masks, mask_path)
    return feats, masks

# ---------------- TRAINING ----------------
def train_model(train_feats, train_masks, val_feats, val_masks):
    H = W = IMG_SIZE // PATCH
    model = LinearSegHead(train_feats.shape[-1], NUM_CLASSES, H, W).to(DEVICE)

    weights = compute_class_weights(train_masks)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scaler = torch.amp.GradScaler(device_type="cuda")

    best_iou = 0.0
    history = []

    for epoch in range(EPOCHS):
        model.train()
        for f, m in DataLoader(
            TensorDataset(train_feats, train_masks),
            BATCH_SIZE, shuffle=True
        ):
            f, m = f.to(DEVICE), m.to(DEVICE)
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                out = model(f)
                out = F.interpolate(out, m.shape[1:], mode="bilinear", align_corners=False)
                loss = criterion(out, m)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # Validation
        model.eval()
        ious = []
        with torch.no_grad():
            for f, m in DataLoader(TensorDataset(val_feats, val_masks), BATCH_SIZE):
                f, m = f.to(DEVICE), m.to(DEVICE)
                out = model(f)
                out = F.interpolate(out, m.shape[1:], mode="bilinear", align_corners=False)
                preds = out.argmax(1)
                ious.append(compute_iou(preds.cpu(), m.cpu(), NUM_CLASSES))

        val_iou = np.mean(ious)
        history.append(val_iou)
        print(f"Epoch [{epoch+1}/{EPOCHS}] | Val IoU: {val_iou:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), f"{SAVE_DIR}/best_model.pth")

    plt.plot(history)
    plt.xlabel("Epoch")
    plt.ylabel("Val IoU")
    plt.savefig(f"{SAVE_DIR}/val_curve.png")
    plt.close()

    return model

# ---------------- TEST ----------------
@torch.no_grad()
def evaluate(model, feats, masks):
    model.eval()
    ious = []
    for f, m in DataLoader(TensorDataset(feats, masks), BATCH_SIZE):
        f, m = f.to(DEVICE), m.to(DEVICE)
        out = model(f)
        out = F.interpolate(out, m.shape[1:], mode="bilinear", align_corners=False)
        preds = out.argmax(1)
        ious.append(compute_iou(preds.cpu(), m.cpu(), NUM_CLASSES))
    return np.mean(ious)

# ---------------- CLASS-WISE IOU ----------------
@torch.no_grad()
def per_class_iou(model, feats, masks):
    model.eval()

    total_inter = torch.zeros(NUM_CLASSES)
    total_union = torch.zeros(NUM_CLASSES)

    for f, m in DataLoader(TensorDataset(feats, masks), BATCH_SIZE):
        f, m = f.to(DEVICE), m.to(DEVICE)

        out = model(f)
        out = F.interpolate(out, m.shape[1:], mode="bilinear", align_corners=False)
        preds = out.argmax(1)

        for c in range(NUM_CLASSES):
            total_inter[c] += ((preds == c) & (m == c)).sum().cpu()
            total_union[c] += ((preds == c) | (m == c)).sum().cpu()

    return (total_inter / (total_union + 1e-6)).numpy()

def plot_per_class_iou(class_ious):
    plt.figure(figsize=(10, 4))
    bars = plt.bar(CLASS_NAMES, class_ious)

    plt.ylabel("IoU")
    plt.xlabel("Class")
    plt.title("Per-Class IoU (Test Set)")
    plt.ylim(0, 1)
    plt.xticks(rotation=30, ha="right")
    plt.grid(axis="y", alpha=0.3)

    for bar, iou in zip(bars, class_ious):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.02,
            f"{iou:.2f}",
            ha="center",
            va="bottom",
            fontsize=9
        )

    plt.tight_layout()
    plt.savefig(f"{SAVE_DIR}/per_class_iou.png")
    plt.show()

# ---------------- VISUALIZE ----------------
def decode_mask(mask):
    """Convert class IDs to RGB image"""
    h, w = mask.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for c in range(NUM_CLASSES):
        rgb[mask == c] = COLORS[c]
    return rgb

def create_error_map(pred_mask, gt_mask):
    h, w = gt_mask.shape
    error_map = np.zeros((h, w, 3), dtype=np.uint8)

    correct = pred_mask == gt_mask
    incorrect = pred_mask != gt_mask

    error_map[correct] = [0, 255, 0]  # Green
    error_map[incorrect] = [255, 0, 0]  # Red
    return error_map

@torch.no_grad()
def visualize_predictions(model, dataset, feats, num_samples=3):
    model.eval()

    for i in range(num_samples):
        img, gt_mask = dataset[i]
        feat = feats[i].unsqueeze(0).to(DEVICE)

        out = model(feat)
        out = F.interpolate(out, size=gt_mask.shape, mode="bilinear", align_corners=False)
        pred = out.argmax(1).squeeze().cpu().numpy()

        img_np = img.permute(1, 2, 0).cpu().numpy()
        img_np = (img_np - img_np.min()) / (img_np.max() - img_np.min())

        gt_rgb = decode_mask(gt_mask.numpy())
        pred_rgb = decode_mask(pred)
        error_rgb = create_error_map(pred, gt_mask.numpy())

        plt.figure(figsize=(16, 4))

        plt.subplot(1, 4, 1)
        plt.title("Image")
        plt.imshow(img_np)
        plt.axis("off")

        plt.subplot(1, 4, 2)
        plt.title("Ground Truth")
        plt.imshow(gt_rgb)
        plt.axis("off")

        plt.subplot(1, 4, 3)
        plt.title("Prediction")
        plt.imshow(pred_rgb)
        plt.axis("off")

        plt.subplot(1, 4, 4)
        plt.title("Error Map")
        plt.imshow(error_rgb)
        plt.axis("off")

        plt.tight_layout()
        plt.savefig(f"{SAVE_DIR}/prediction_{i}.png")
        plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    backbone = torch.hub.load(
        "facebookresearch/dinov2", "dinov2_vits14"
    ).to(DEVICE).eval()

    train_ds = SegDataset(ROOT, "train")
    val_ds = SegDataset(ROOT, "val")
    test_ds = SegDataset(ROOT, "test")

    train_feats, train_masks = cache_features(
        backbone, DataLoader(train_ds, BATCH_SIZE), "train"
    )
    val_feats, val_masks = cache_features(
        backbone, DataLoader(val_ds, BATCH_SIZE), "val"
    )
    test_feats, test_masks = cache_features(
        backbone, DataLoader(test_ds, BATCH_SIZE), "test"
    )

    model = train_model(train_feats, train_masks, val_feats, val_masks)
    model.load_state_dict(torch.load(f"{SAVE_DIR}/best_model.pth"))

    test_iou = evaluate(model, test_feats, test_masks)
    print(f"🔥 FINAL TEST IoU: {test_iou:.4f}")

    class_ious = per_class_iou(model, test_feats, test_masks)
    plot_per_class_iou(class_ious)

    visualize_predictions(model, test_ds, test_feats, num_samples=3)
