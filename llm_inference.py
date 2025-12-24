import re
from groq import Groq
from constants import (
    GROQ_API_KEY,
    LLM_MODEL_NAME,
    LLM_TEMPERATURE,
    LLM_MAX_TOKENS,
)

client = Groq(api_key=GROQ_API_KEY)

class LLMQuotaError(Exception):
    """Custom exception for LLM quota / rate limit errors."""
    pass

def build_prompt(post):
    return f"""
You are a content safety evaluator.
Rate the EMOTIONAL DISTRESS level of the following post on a scale from 0 to 1.

Return ONLY a number between 0 and 1.

Post:
\"\"\"{post}\"\"\"

Severity score:
"""

def get_llm_score(post):
    try: 
        prompt = build_prompt(post)

        response = client.chat.completions.create(
            model=LLM_MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=10,
            temperature=0
        )

        raw = response.choices[0].message.content.strip()
        match = re.search(r"([0-9]*\.?[0-9]+)", raw)
        if not match:
            return 0.0

        return max(0.0, min(1.0, float(match.group(1))))
    
    except Exception as e:
        raise LLMQuotaError(
            "Out of requests for the day. Please try again later."
        ) from e


def llm_score_batch(posts):
    return [get_llm_score(p) for p in posts]
