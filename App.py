import streamlit as st
from transformers import pipeline
import torch

# --- 1. Load Zero-Shot Classifier ---
@st.cache_resource(show_spinner=True)
def load_model():
    classifier = pipeline(
        "zero-shot-classification",
        model="MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        device=0 if torch.cuda.is_available() else -1,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    )
    return classifier

classifier = load_model()

# --- 2. Candidate Labels ---
candidate_labels = [
    "shopping for software or tools",           # High Intent
    "hiring employees or growing team",         # Medium Intent
    "discussing HR trends or articles",         # Low Intent
    "off-topic, greeting, hate, spam, personal" # No Intent
]

# --- 3. Streamlit UI ---
st.title("🔥 Ultimate Intent Classifier 2025")
st.markdown("""
Powered by **DeBERTa-v3 zero-shot** model.  
Type your text below or select an example from the dropdown to see the predicted intent.
""")

# Example sentences (15+)
examples = [
    "Looking for HR automation tools",
    "Hiring an HR Manager",
    "Top HR trends in 2025",
    "I love playing football",
    "Need a recruitment AI solution",
    "Our company is expanding the dev team",
    "Best productivity apps for managers",
    "Looking to buy payroll software",
    "We are recruiting a data analyst",
    "AI tools for office automation",
    "Good morning everyone!",
    "Company culture tips for employees",
    "Hiring interns for marketing department",
    "Off-topic message about hobbies",
    "Searching for employee engagement platforms"
]

# --- Input ---
text_input = st.text_area("Type your text here:")
selected_example = st.selectbox("Or choose an example:", ["--None--"] + examples)

# Use either input or selected example
text_to_predict = text_input.strip() if text_input.strip() else None
if selected_example != "--None--":
    text_to_predict = selected_example

# --- Prediction ---
if text_to_predict:
    result = classifier(text_to_predict, candidate_labels, multi_label=False)
    top_label = result["labels"][0]
    score = result["scores"][0]

    # Map to our categories
    if "shopping" in top_label:
        icon, tag = "HIGH", "HIGH INTENT"
    elif "hiring" in top_label:
        icon, tag = "MEDIUM", "MEDIUM INTENT"
    elif "discussing" in top_label:
        icon, tag = "LOW", "LOW INTENT"
    else:
        icon, tag = "NONE", "NO INTENT"

    # Display
    st.markdown("### Prediction Result")
    st.write(f"**Intent:** {tag} ({icon})")
    st.write(f"**Confidence:** {score:.1%}")
    st.write(f"**Matched Label:** {top_label}")
