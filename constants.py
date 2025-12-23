# =========================
# MODEL CONFIGURATION
# =========================

# Path to fine-tuned HuggingFace model
FINETUNED_MODEL_PATH = "./distilbert_goemotions"

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
    "grief": 1.0,
    "fear": 0.9,
    "anger": 0.85,
    "remorse": 0.8,
    "disgust": 0.75,
    "sadness": 0.7,
    "nervousness": 0.65,
    "disappointment": 0.6,
    "embarrassment": 0.55,
    "disapproval": 0.4,
    "annoyance": 0.3,
}