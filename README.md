# Emotional Distress Severity Scoring Demo

This project provides a **Streamlit-based demo** for scoring the **emotional distress severity** of social media posts or text feeds on a continuous scale from **0 to 1**.

The demo allows users to compare:
- **LLM Prompting** (API-based inference)
- **Fine-tuned Transformer Model** (hosted on Hugging Face)

The system supports **single posts or multiple newline-separated feeds** and outputs a severity score for each post.

---

## 🔍 Project Overview

Emotional distress detection is important for content moderation and safety-aware recommendation systems.  
This project explores two approaches:

1. **LLM Prompting**  
   Uses a large language model with carefully designed prompts to directly predict distress severity.

2. **Fine-tuned Model**  
   A DistilBERT model fine-tuned on the **GoEmotions** dataset, with a custom aggregation scheme to map multi-label emotion predictions to a single severity score.

The Streamlit app provides a simple interface to compare these approaches interactively.

---

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
```

### Run the app
```
streamlit run app.py
```

## Author

Vishakha Mistry