import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from constants import (
    FINETUNED_MODEL_PATH,
    MAX_SEQ_LENGTH,
    DEVICE,
    SEVERITY_ACTIVATION_THRESHOLD,
    EMOTION_WEIGHTS,
)

DEVICE = DEVICE if torch.cuda.is_available() else "cpu"

# Load model
tokenizer = AutoTokenizer.from_pretrained(FINETUNED_MODEL_PATH)
model = AutoModelForSequenceClassification.from_pretrained(FINETUNED_MODEL_PATH)
model.to(DEVICE)
model.eval()

emotion_labels = model.config.id2label.values()

weights = np.array([EMOTION_WEIGHTS.get(lab, 0.0) for lab in emotion_labels])
max_weight = max(weights.max(), 1e-6)

def compute_severity(probs):
    t = SEVERITY_ACTIVATION_THRESHOLD
    activated = np.where(probs > t, (probs - t) / (1 - t), 0.0)
    score = (activated * weights).sum() / max_weight
    return float(np.clip(score, 0.0, 1.0))

@torch.no_grad()
def finetuned_score_batch(posts):
    enc = tokenizer(
        posts,
        padding=True,
        truncation=True,
        max_length=MAX_SEQ_LENGTH,
        return_tensors="pt"
    ).to(DEVICE)

    logits = model(**enc).logits
    probs = torch.sigmoid(logits).cpu().numpy()

    return [compute_severity(p) for p in probs]
