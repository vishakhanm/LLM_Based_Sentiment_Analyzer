# =========================
# MODEL CONFIGURATION
# =========================

# Path to fine-tuned HuggingFace model
FINETUNED_MODEL_PATH = "vishakhanm/distilbert-goemotions-distress"

# Max sequence length for tokenizer
MAX_SEQ_LENGTH = 128

# Device configuration
DEVICE = "cuda"  # will auto-fallback to cpu if unavailable


# =========================
# LLM CONFIGURATION
# =========================

# Groq API key (recommend using env vars in production)
GROQ_API_KEY = "YOUR_GROQ_API_KEY"

# LLM model name
LLM_MODEL_NAME = "llama-3.3-70b-versatile"

# LLM generation settings
LLM_TEMPERATURE = 0.0
LLM_MAX_TOKENS = 10


# =========================
# SEVERITY SCORING
# =========================

# Probability activation threshold
SEVERITY_ACTIVATION_THRESHOLD = 0.05

# Emotion → severity weight mapping
EMOTION_WEIGHTS = {
    "LABEL_16": 1.0, #grief
    "LABEL_14": 0.9, #fear
    "LABEL_2": 0.85, #anger
    "LABEL_24": 0.8, #remorse
    "LABEL_11": 0.75, #disgust
    "LABEL_25": 0.7, #sadness
    "LABEL_19": 0.65, #nervousness
    "LABEL_9": 0.6, #disappointment
    "LABEL_12": 0.55, #embarrassment
    "LABEL_10": 0.4, #disapproval
    "LABEL_3": 0.3, #annoyance
}

# Feed-level aggregation
FEED_SEVERITY_METHOD = "top_k_mean"   # options: "mean", "max", "top_k_mean"
FEED_TOP_K_RATIO = 0.4          # used only if method = "top_k_mean"

# Risk flag threshold
FEED_SEVERITY_THRESHOLD = 0.6