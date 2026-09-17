# 💬 Medical Visual Question Answering (VQA) — Multimodal AI

> **Author:** Mahnoor ([@mahnoor-2722](https://github.com/mahnoor-2722))  
> **Domain:** Multimodal Deep Learning · Vision + Language · Healthcare AI  
> **Dataset:** VQA-RAD (Radiology Visual Question Answering Dataset)

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red?logo=pytorch)
![Hugging Face](https://img.shields.io/badge/HuggingFace-Model%20Hosted-yellow?logo=huggingface)
![Streamlit](https://img.shields.io/badge/Streamlit-Live%20Demo-ff4b4b?logo=streamlit)
![License](https://img.shields.io/badge/License-MIT-green)



## 🌐 Live Demos & Resources

🔗 **[Try the Live Streamlit Multimodal Web App](https://SHARE-STREAMLIT-URL.streamlit.app)**  
🔗 **[Fine-Tuned BLIP Model on Hugging Face Hub](https://huggingface.co/mahnoor-2722/blip-medical-vqa-rad)**



## 📌 Project Overview

This project implements an end-to-end **Medical Visual Question Answering (VQA)** system capable of interpreting clinical radiology images (X-Rays, MRIs, CT scans) and answering natural language diagnostic questions in real time.

Unlike standard Computer Vision models that only process static images, this system fuses **Vision Transformers (ViT)** with **Language Decoders (LLM heads)** through **Cross-Attention mechanisms** to reason across both visual and textual domains simultaneously.



## 🏗️ Architecture & Multimodal Pipeline

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│                           MULTIMODAL BLIP PIPELINE                          │
├─────────────────────────────────────────────────────────────────────────────┤
│  [📷 Medical Scan] ──────►  Vision Transformer (ViT-B)                      │
│                                    │                                        │
│                                    ▼                                        │
│                             [Cross-Attention] ◄── [💬 Clinical Question]    │
│                                    │                                        │
│                                    ▼                                        │
│                         Text Decoder (LLM Head)                             │
│                                    │                                        │
│                                    ▼                                        │
│                        [🤖 Generated Clinical Answer]                        │
└─────────────────────────────────────────────────────────────────────────────┘

## Task Formulation
This project implements **Clinical Diagnostic Screening VQA**:
- Input: medical image + clinical question
- Output: yes/no finding prediction

The fine-tuned model specializes in closed-ended clinical queries
(e.g., fracture/integrity/effusion/cardiomegaly style screening).

For open-ended descriptive questions, the app can fall back to base BLIP.