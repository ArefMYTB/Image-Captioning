# Image Captioning with Transformers (Encoder–Decoder)

This project implements an **image captioning system** using:

- **ResNet50** (pretrained) as an **encoder** for visual features  
- A **custom Transformer decoder** for caption generation  
- **Beam search** and **verification steps** for better inference  

---

## Features

- **Pretrained ResNet50 Encoder**  
  Extracts strong spatial feature maps (without training a vision backbone).  
  → Faster convergence, better captions.

- **Patch/Region Features**  
  CNN feature maps are flattened into a sequence so the decoder can attend to different image regions.

- **Transformer Decoder**  
  Autoregressive caption generation with **self-attention** (causal) and **cross-attention** to image features.

- **Beam Search**  
  Improves over greedy decoding by exploring top-`k` candidate captions.

- **Evaluation Metrics**  
  - **BLEU** (via `nltk`)  
  - **CIDEr** (via `pycocoevalcap`) → correlates better with human judgement.

- **Tokenizer**: `distilbert-base-uncased`  
  - Converts captions ↔ token IDs  
  - On-the-fly tokenization (no need to pre-tokenize the dataset)  
  - **TokenDrop** regularization: randomly replaces tokens with `<pad>` to reduce teacher-forcing memorization.

---

## Model Architecture

- **Encoder: ResNet50**  
  - Use intermediate feature map (e.g., `layer4`) with shape `[C, H', W']`  
  - Flatten spatial grid into a sequence `[H'·W', C]`.

- **Decoder: Transformer**  
  - Inputs = token embeddings + sinusoidal positional embeddings  
  - Each block:  
    1. Self-attention (causal mask)  
    2. Cross-attention (to encoder features)  
    3. MLP  

- **Output**  
  - Final logits map to tokenizer’s vocabulary.  
  - Decoder learns to attend to different regions per token.

---

## Inference Enhancements

I include **verified caption generation** that combines **beam search**, **object detection**, **language model scoring**, and **hallucination checks**:

1. **Object Detection**: via Faster R-CNN (pretrained on COCO) and YOLOv8.  
2. **Beam Search Candidate Generation**: multiple possible captions.  
3. **LM Scoring (DistilGPT-2)**: prefer grammatically fluent captions.  
4. **Object-Match Scoring**: penalize hallucinated objects.  
5. **N-gram Repeat Removal**: eliminate repetitive phrases.  
6. **Final Selection**: best caption chosen via weighted scoring.

---

## Example: Verified Caption Generation Improves Reliability

| Image | Reference Caption | Model Output (Verified) |
|-------|------------------|--------------------------|
| <img src="https://github.com/user-attachments/assets/b54cbe50-64b0-43a5-be6c-550d36ce0093" width="250"/> | a male in a white shirt and black shorts playing tennis | a man holding a tennis racket on the tennis court |

---
**Why this matters:**  
- Both captions are **plausible and correct**.  
- Without verification, models may hallucinate objects or produce ungrammatical text.  
- The verified approach ensures captions remain **accurate, fluent, and grounded in the image**.
