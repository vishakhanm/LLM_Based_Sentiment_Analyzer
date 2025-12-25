import streamlit as st
from llm_inference import llm_score_batch
from finetuned_inference import finetuned_score_batch
from utilities import aggregate_feed_severity

st.set_page_config(page_title="Distress Severity Demo", layout="centered")

st.title("🧠 Emotional Distress Severity Scoring")
st.markdown(
    "Choose a model, enter one or more posts (newline-separated), and get severity scores (0–1)."
)

# --- Model selection ---
model_choice = st.radio(
    "Select Model",
    options=["LLM Prompting", "Fine-tuned Model"],
    horizontal=True
)

# --- Text input ---
input_text = st.text_area(
    "Enter post(s)",
    placeholder="Enter one post per line...\n\nExample:\nI feel so exhausted lately.\nNothing seems to work out.",
    height=200
)

# --- Run button ---
if st.button("Run Model"):
    if not input_text.strip():
        st.warning("Please enter at least one post.")
    else:
        posts = [p.strip() for p in input_text.split("\n") if p.strip()]

        with st.spinner("Scoring posts..."):
            if model_choice == "LLM Prompting":
                scores = llm_score_batch(posts)
            else:
                scores = finetuned_score_batch(posts)

        # st.success("Done!")
        feed_severity, is_flagged = aggregate_feed_severity(scores)

        # --- Output ---
        st.subheader("Feed-Level Result")
        st.markdown(f"**Feed Severity Score:** `{feed_severity:.3f}`")

        if is_flagged:
            st.error("⚠️ High distress feed detected (above threshold).")
        else:
            st.success("✅ Feed distress level is within acceptable range.")
            
        st.subheader("Post-wise severity")
        for post, score in zip(posts, scores):
            st.markdown(f"**Post:** {post}")
            st.markdown(f"**Severity Score:** `{score:.3f}`")
            st.divider()
