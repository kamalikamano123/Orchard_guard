import json
from pathlib import Path
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm

# ========================
# Paths and Config
# ========================
ROOT = Path("..").resolve()
DATA = ROOT / "dataset"
SAVE = ROOT / "model"
SAVE.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
EPOCHS = 5  # Train longer
LR = 1e-4

# ========================
# Data Augmentation
# ========================
train_tfms = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.RandomResizedCrop(299),   # Larger crop
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(0.3,0.3,0.3),
    transforms.RandomAffine(10),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])
eval_tfms = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

train_ds = datasets.ImageFolder(DATA/"train", transform=train_tfms)
val_ds   = datasets.ImageFolder(DATA/"val",   transform=eval_tfms)
test_ds  = datasets.ImageFolder(DATA/"test",  transform=eval_tfms)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, pin_memory=True)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)

class_names = train_ds.classes
with open(SAVE/"class_names.json","w") as f:
    json.dump(class_names,f)

# ========================
# Model - ResNet50 Fine-Tuned
# ========================
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)

# Freeze all layers except last block (layer4) + FC
for name, param in model.named_parameters():
    if "layer4" in name or "fc" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(model.fc.in_features, len(class_names))
)
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # Decay LR every 10 epochs

# ========================
# Training Function
# ========================
def run_epoch(loader, train=True):
    model.train(train)
    total, correct, running_loss = 0, 0, 0.0
    pbar = tqdm(loader)
    for x,y in pbar:
        x, y = x.to(DEVICE), y.to(DEVICE)
        if train: optimizer.zero_grad()
        out = model(x)
        loss = criterion(out,y)
        if train:
            loss.backward()
            optimizer.step()
        running_loss += loss.item()*x.size(0)
        pred = out.argmax(1)
        correct += (pred==y).sum().item()
        total += y.size(0)
        pbar.set_description(f"{'Train' if train else 'Val'} loss {running_loss/total:.4f} acc {correct/total:.4f}")
    return running_loss/total, correct/total

# ========================
# Training Loop
# ========================
best_acc, best_path = 0.0, SAVE/"apple_resnet50_best.pth"
patience, patience_counter = 3, 0   # stop if no improvement for 3 epochs

for e in range(1, EPOCHS+1):
    print(f"\nEpoch {e}/{EPOCHS}")
    run_epoch(train_loader, True)
    _, va_acc = run_epoch(val_loader, False)

    # save best model
    if va_acc > best_acc:
        best_acc = va_acc
        torch.save(model.state_dict(), best_path)
        print(f"📦 Saved best → {best_path} (val acc={va_acc:.4f})")
        patience_counter = 0  # reset patience
    else:
        patience_counter += 1
        print(f"⚠️ No improvement for {patience_counter} epoch(s)")

    # early stopping
    if patience_counter >= patience:
        print("🛑 Early stopping triggered! No improvement in validation accuracy.")
        break

# ✅ Load best model and evaluate on test set
model.load_state_dict(torch.load(best_path, map_location=DEVICE))
model.eval()
y_true, y_pred = [], []
with torch.no_grad():
    for x,y in tqdm(test_loader, desc="Test"):
        out = model(x.to(DEVICE))
        y_pred += out.argmax(1).cpu().tolist()
        y_true += y.tolist()

print("\nClassification report:")
print(classification_report(y_true, y_pred, target_names=class_names))
print("Confusion matrix:")
print(confusion_matrix(y_true, y_pred))
