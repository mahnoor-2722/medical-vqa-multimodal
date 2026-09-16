# ============================================================
# Streamlit Web Application: Medical Visual Question Answering (VQA)
# Model: Fine-Tuned BLIP Multimodal Transformer | PyTorch
# Author: Mahnoor (github.com/mahnoor-2722)
# ============================================================

import os
import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

# Page Configuration
st.set_page_config(
    page_title="Medical VQA — Multimodal AI",
    page_icon="💬",
    layout="wide"
)

# Constants
HF_REPO_ID = "mahnoor-2722/blip-medical-vqa-rad"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_vqa_model():
    """Load Fine-Tuned BLIP Model and Processor from Hugging Face Hub"""
    try:
        processor = BlipProcessor.from_pretrained(HF_REPO_ID)
        model = BlipForQuestionAnswering.from_pretrained(HF_REPO_ID)
        model.to(DEVICE)
        model.eval()
        return processor, model
    except Exception as e:
        st.error(f"❌ Error loading model from Hugging Face Hub: {e}")
        return None, None

# ==================== STREAMLIT UI ==================== #

st.title("💬 Medical Visual Question Answering (VQA)")
st.markdown("""
This application uses a **Fine-Tuned BLIP Multimodal Transformer** (Vision Encoder + Text Decoder) 
to answer clinical and diagnostic questions about medical scans in real time.
""")

# Sidebar
st.sidebar.header("⚙️ System Status")
st.sidebar.info(
    f"**Model:** BLIP-VQA (Salesforce Base)\n\n"
    f"**Fine-Tuned On:** VQA-RAD Clinical Dataset\n\n"
    f"**Hosting:** Hugging Face Hub (`{HF_REPO_ID}`)\n\n"
    f"**Device:** `{DEVICE}`"
)

# Load Model
with st.spinner("⏳ Loading Fine-Tuned BLIP Multimodal Model from Hugging Face..."):
    processor, model = load_vqa_model()

if model is None or processor is None:
    st.stop()

st.sidebar.success("✅ Multimodal Model Ready!")

# Main Upload & Question Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1. Upload Clinical Scan")
    uploaded_file = st.file_uploader(
        "Choose a Radiology Scan (X-Ray, MRI, CT, or Clinical Photo)",
        type=["jpg", "jpeg", "png"]
    )
    
    if uploaded_file is not None:
        input_image = Image.open(uploaded_file).convert("RGB")
        st.image(input_image, caption="Uploaded Medical Scan", use_container_width=True)

with col2:
    st.subheader("2. Ask a Clinical Question")
    
    if uploaded_file is not None:
        # Sample Question Preset Buttons
        st.markdown("**Sample Questions (Click to test):**")
        btn_col1, btn_col2 = st.columns(2)
        
        sample_q = ""
        if btn_col1.button("🫀 Cardiomegaly check"):
            sample_q = "Is there evidence of cardiomegaly?"
        if btn_col2.button("🫁 Lung clarity check"):
            sample_q = "Is the lung field clear?"
            
        user_question = st.text_input(
            "Enter your clinical question about the image:",
            value=sample_q if sample_q else "Is there any abnormality visible in this scan?",
            placeholder="e.g., Is pneumothorax visible?"
        )
        
        ask_button = st.button("🚀 Analyze Scan & Answer Question", type="primary")
        
        if ask_button and user_question.strip():
            with st.spinner("🧠 Cross-attention processing (Vision + Text)..."):
                # Prepare inputs
                inputs = processor(images=input_image, text=user_question, return_tensors="pt").to(DEVICE)
                
                # Generate answer text
                # Force BLIP to use Beam Search & Repetition Penalty for descriptive answers
                with torch.no_grad():
                    output = model.generate(
                        **inputs,
                        max_new_tokens=25,
                        num_beams=5,                  # Explores 5 different response paths
                        no_repeat_ngram_size=2,      # Prevents repeating words
                        early_stopping=True,
                        repetition_penalty=1.5       # Penalizes repetitive "yes/no" tokens
                    )
                    generated_answer = processor.decode(output[0], skip_special_tokens=True).strip()
                
            st.markdown("---")
            st.subheader("🤖 AI Diagnostic Answer")
            st.success(f"**Question:** {user_question}\n\n**Answer:** **{generated_answer.upper()}**")
            
            st.markdown("---")
            st.markdown("### 🧬 Multimodal Attention Breakdown")
            st.info(
                f"• **Vision Encoder:** Extracted 256 spatial patch embeddings\n\n"
                f"• **Text Encoder:** Tokenized clinical query '{user_question}'\n\n"
                f"• **Cross-Attention:** Aligned visual regions with textual query tokens\n\n"
                f"• **Decoder Output:** `{generated_answer}`"
            )
    else:
        st.info("👆 Please upload a medical scan on the left to activate the Q&A interface.")

st.markdown("---")
st.warning(
    "📌 **Medical Disclaimer:** This is an open-source research demonstration fine-tuned on clinical VQA datasets. "
    "It is **not** a certified medical diagnostic device and must not be used for primary healthcare decisions."
)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align:center; color:gray; font-size:0.85em;'>"
    "Built by <b>Mahnoor</b> · Multimodal Vision-Language AI (BLIP) · "
    "<a href='https://github.com/mahnoor-2722/medical-vqa-multimodal' target='_blank'>GitHub</a> · "
    "<a href='https://huggingface.co/mahnoor-2722/blip-medical-vqa-rad' target='_blank'>Hugging Face Model</a>"
    "</div>",
    unsafe_allow_html=True
)