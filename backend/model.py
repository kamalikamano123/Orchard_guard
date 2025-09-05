import os, json, torch
from PIL import Image
from torchvision import transforms, models
from torch import nn

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "model")
DEVICE = "cpu"  # use CPU for portability (set to "cuda" if deploying on GPU)

# Load label map
with open(os.path.join(MODEL_DIR, "class_names.json")) as f:
    CLASS_NAMES = json.load(f)

# Same preprocessing used during training
_eval_tfms = transforms.Compose([
    transforms.Resize((384,384)),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406],
                         [0.229,0.224,0.225]),
])

# Build ResNet50 model and load weights
_model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
_model.fc = nn.Sequential(
    nn.Dropout(0.4),
    nn.Linear(_model.fc.in_features, len(CLASS_NAMES))
)
_state = torch.load(os.path.join(MODEL_DIR, "apple_resnet50_best.pth"), map_location=DEVICE)
_model.load_state_dict(_state)
_model.eval()

def predict_pil(img: Image.Image):
    """Predict apple leaf disease from a PIL image"""
    x = _eval_tfms(img.convert("RGB")).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        out = _model(x)
        prob = torch.softmax(out, dim=1)[0]
        idx = int(torch.argmax(prob))
    return {
        "label": CLASS_NAMES[idx],
        "confidence": float(prob[idx])
    }
