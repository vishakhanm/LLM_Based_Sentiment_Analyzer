# Emotional Distress Severity Scoring Demo

Emotional distress detection is important for content moderation and safety-aware recommendation systems. This project provides a **Streamlit-based demo** for scoring the **emotional distress severity** of social media posts or text feeds on a continuous scale from **0 to 1**. The system supports **feed-level emotional distress analysis** when multiple posts are provided as input.

## Feed Aggregation Logic

When the user inputs multiple newline-separated posts:
1. Each post is scored individually for emotional distress (∈ [0, 1]).
2. Individual post scores are aggregated into a single **feed severity score** using a configurable strategy:
   - `mean`: Average severity across all posts
   - `max`: Maximum severity among posts
   - `top_k_mean`: Mean of the top-k most severe posts

Aggregation behavior is configurable via constants.

## Threshold-Based Risk Flagging

After aggregation:
- The feed is **flagged** if the aggregated severity score exceeds a predefined threshold.
- This enables identification of **high-risk feeds** even when individual posts may vary in severity.

---

## Approach
This project explores two approaches:

1. **LLM Prompting**  
   Uses a large language model with carefully designed prompts to directly predict distress severity.

2. **Fine-tuned Model**  
   A DistilBERT model fine-tuned on the **GoEmotions** dataset, with a custom aggregation scheme to map multi-label emotion predictions to a single severity score.

## 🧠 Models Used

### 1. Fine-tuned Model
- Base model: `distilbert-base-uncased`
- Dataset: GoEmotions
- Task: Multi-label emotion classification
- Output: Emotion probabilities → weighted severity score ∈ [0, 1]
- Hosting: **Hugging Face Model Hub**

### 2. LLM Prompting
- Model: LLaMA (via Groq API)
- Method: Zero-shot prompting
- Output: Direct severity score ∈ [0, 1]

---

## 🖥️ Demo Features

- Choose between **LLM Prompting** or **Fine-tuned Model**
- Input:
  - Single post
  - Multiple posts (one per line)
- Output:
  - Per-post emotional distress severity score
  - Feed level emotional distress severity score
- Graceful handling of API rate limits for LLM inference

---

## Setup Instructions

### Clone the Repository

```bash
git clone <your-repo-url>
cd project
```

### Install dependencies
```
pip install -r requirements.txt
```

### Edit constants.py
```
GROQ_API_KEY = "YOUR_GROQ_API_KEY"
FEED_SEVERITY_METHOD = "mean"   # mean | max | top_k_mean
FEED_TOP_K_RATIO = 0.2          # used if method = top_k_mean
FEED_SEVERITY_THRESHOLD = 0.6   # flagging threshold
```

### Run the app
```
streamlit run app.py
```

## Author

Vishakha Mistry