import streamlit as st
from transformers import pipeline
import torch

# --- Load Zero-Shot Model ---
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

# --- Candidate Labels ---
candidate_labels = [
    "shopping for software or tools",           
    "hiring employees or growing team",         
    "discussing HR trends or articles",         
    "off-topic, greeting, hate, spam, personal"
]

# --- UI ---
st.title("🔥 Ultimate Intent Classifier 2025")
st.markdown("""
Powered by **DeBERTa-v3 zero-shot** model.  
Type your text or select a predefined example to see the predicted intent.
""")

# Predefined examples
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

# --- Single Input Box ---
text_input = st.text_area("Type your text here (or select an example below):", height=120)

# Dropdown to select example and copy to text_input
selected_example = st.selectbox("Or select an example to fill above:", ["--None--"] + examples)
if selected_example != "--None--" and selected_example != text_input:
    text_input = selected_example

# Predict button
if st.button("Predict"):
    if text_input.strip():
        # Prediction
        result = classifier(text_input.strip(), candidate_labels, multi_label=False)
        top_label = result["labels"][0]
        score = result["scores"][0]

        # Map to HIGH/MEDIUM/LOW/NO
        if "shopping" in top_label:
            icon, tag = "HIGH", "HIGH INTENT"
        elif "hiring" in top_label:
            icon, tag = "MEDIUM", "MEDIUM INTENT"
        elif "discussing" in top_label:
            icon, tag = "LOW", "LOW INTENT"
        else:
            icon, tag = "NONE", "NO INTENT"

        # Display result
        st.markdown("### Prediction Result")
        st.write(f"**Intent:** {tag} ({icon})")
        st.write(f"**Confidence:** {score:.1%}")
        st.write(f"**Matched Label:** {top_label}")
    else:
        st.warning("Please type a sentence or select an example to predict.")
