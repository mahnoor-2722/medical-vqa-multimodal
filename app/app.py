# ============================================================
# Medical VQA App (Hybrid)
# Fine-tuned BLIP = clinical yes/no screening
# Base BLIP = open-ended fallback
# ============================================================

import torch
import streamlit as st
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering

st.set_page_config(page_title="Medical VQA", page_icon="💬", layout="wide")

FT_REPO = "mahnoor-2722/blip-medical-vqa-rad"
BASE_REPO = "Salesforce/blip-vqa-base"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_models():
    ft_processor = BlipProcessor.from_pretrained(FT_REPO)
    ft_model = BlipForQuestionAnswering.from_pretrained(FT_REPO).to(DEVICE).eval()

    base_processor = BlipProcessor.from_pretrained(BASE_REPO)
    base_model = BlipForQuestionAnswering.from_pretrained(BASE_REPO).to(DEVICE).eval()
    return ft_processor, ft_model, base_processor, base_model

def is_closed_ended(q: str) -> bool:
    q = q.lower().strip()
    starters = ("is ", "are ", "was ", "were ", "do ", "does ", "can ", "could ")
    return q.startswith(starters) or q.startswith("is there") or q.startswith("are there")

st.title("💬 Medical Visual Question Answering (VQA)")
st.caption("Hybrid system: clinical yes/no screening (fine-tuned) + open-ended fallback (base BLIP)")

ft_processor, ft_model, base_processor, base_model = load_models()
st.sidebar.success("✅ Models loaded")
st.sidebar.info(f"Device: {DEVICE}")

c1, c2 = st.columns(2)

with c1:
    st.subheader("1) Upload scan")
    f = st.file_uploader("X-ray / MRI / CT / clinical image", type=["jpg", "jpeg", "png"])
    image = None
    if f is not None:
        image = Image.open(f).convert("RGB")
        st.image(image, use_container_width=True)

with c2:
    st.subheader("2) Ask a question")
    if image is None:
        st.info("Upload an image first")
    else:
        st.markdown("**Try these (best for fine-tuned model):**")
        b1, b2 = st.columns(2)
        b3, b4 = st.columns(2)
        preset = ""
        if b1.button("Are the bone structures intact?"):
            preset = "Are the bone structures intact?"
        if b2.button("Is there evidence of cardiomegaly?"):
            preset = "Is there evidence of cardiomegaly?"
        if b3.button("Is pneumothorax visible?"):
            preset = "Is pneumothorax visible?"
        if b4.button("Is pleural effusion detected?"):
            preset = "Is pleural effusion detected?"

        q = st.text_input("Clinical question", value=preset or "Are the bone structures intact?")

        if st.button("Analyze", type="primary"):
            closed = is_closed_ended(q)

            with st.spinner("Running multimodal inference..."):
                if closed:
                    processor, model, mode = ft_processor, ft_model, "Fine-tuned clinical screener"
                else:
                    processor, model, mode = base_processor, base_model, "Base BLIP open-ended fallback"

                inputs = processor(images=image, text=q, return_tensors="pt").to(DEVICE)
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        max_new_tokens=20,
                        num_beams=4,
                        early_stopping=True
                    )
                ans = processor.decode(out[0], skip_special_tokens=True).strip()

            st.markdown("---")
            st.subheader("AI Answer")
            st.success(f"**Mode:** {mode}\n\n**Q:** {q}\n\n**A:** **{ans}**")

            if closed:
                st.info("This question looks closed-ended, so the fine-tuned clinical model was used.")
            else:
                st.warning("Open-ended question detected. Used base BLIP fallback because fine-tuned model is specialized for yes/no clinical screening.")

st.warning("Research demo only — not for clinical diagnosis.")