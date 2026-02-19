# 🏥 Medical Q&A Chatbot (Fine-Tuning Flan-T5-Base with LoRA)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR-GITHUB-USERNAME/medical-chatbot/blob/main/medical_chatbot_Assistant.ipynb)
[![Live Demo](https://img.shields.io/badge/Live%20Demo-HuggingFace%20Spaces-blue?logo=huggingface)](https://huggingface.co/spaces/Numwali/medical-chatbot-lora)
[![Model](https://img.shields.io/badge/Model-Numwali%2Fmedical--chatbot--lora-yellow?logo=huggingface)](https://huggingface.co/Numwali/medical-chatbot-lora)

A domain-specific medical question-answering assistant built by fine-tuning **google/flan-t5-base** with **LoRA (Low-Rank Adaptation)** on 1,800 clinical flashcard Q&A pairs. The fine-tuned model achieves a **+613.5% improvement in ROUGE-L** over the unmodified base model and is permanently deployed on Hugging Face Spaces.

---

## Live Demo

Try the chatbot right now, no setup, no installation needed:

👉 **[https://huggingface.co/spaces/Numwali/medical-chatbot-lora](https://huggingface.co/spaces/Numwali/medical-chatbot-lora)**

---

## Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model and Fine-Tuning](#model-and-fine-tuning)
- [Experiment Results](#experiment-results)
- [Performance Metrics](#performance-metrics)
- [Example Conversations](#example-conversations)
- [How to Run](#how-to-run)
- [Repository Structure](#repository-structure)
- [Requirements](#requirements)

---

## Project Overview

Access to reliable medical information is a critical global challenge, especially in regions where trained healthcare professionals are scarce. This project builds a chatbot that provides clear, concise, and clinically grounded answers to health-related questions, making it useful for medical students revising for exams, caregivers, and people in low-resource settings.

The model was trained in under 10 minutes on a free Google Colab T4 GPU using parameter-efficient fine-tuning, and is hosted permanently on Hugging Face Spaces so anyone can use it without any setup.

---

## Dataset

**Source:** [medalpaca/medical_meadow_medical_flashcards](https://huggingface.co/datasets/medalpaca/medical_meadow_medical_flashcards)

33,955 medical question-answer pairs derived from real clinical flashcard materials used in medical education worldwide. A subset of 2,000 examples was used for this project.

| Property | Value |
|---|---|
| Full dataset size | 33,955 examples |
| Training examples | 1,800 (shuffled, seed=42) |
| Evaluation examples | 200 |
| Average input length | 14.7 words (max: 47) |
| Average output length | 52.9 words (max: 211) |

**Preprocessing pipeline:**

Each input was reformatted with the instruction template `"Answer this medical question: <question>"` to prime the model to behave as an instruction-following assistant. Inputs were tokenised using the Flan-T5 SentencePiece tokeniser, truncated to 256 tokens for inputs and 128 tokens for targets. Padding tokens in label sequences were replaced with -100 so the cross-entropy loss ignores them. Dynamic batch padding was applied using `DataCollatorForSeq2Seq` to reduce memory usage during training.

---

## Model and Fine-Tuning

**Base model:** `google/flan-t5-base`, 250M parameters, encoder-decoder architecture, already instruction-tuned on 1,800+ NLP tasks.

**Method:** LoRA via Hugging Face `peft` inserts small trainable matrices into the attention layers of the frozen base model. Only 1.41% of parameters are actually updated during training.

| LoRA Config | Value |
|---|---|
| Target modules | query and value projections |
| Best rank (r) | 32 |
| Alpha | 64 |
| Trainable parameters | 3,538,944 / 251,116,800 (1.41%) |
| Precision | float32 (fp16 disabled caused gradient underflow) |

---

## Experiment Results

Four controlled experiments were run, each varying the learning rate, LoRA rank, and number of epochs while keeping everything else fixed (batch size 4, gradient accumulation 4, warmup ratio 0.1, weight decay 0.01).

| Experiment | LR | r | Epochs | ROUGE-1 | ROUGE-2 | ROUGE-L | BLEU | Time | GPU |
|---|---|---|---|---|---|---|---|---|---|
| Base (no fine-tuning) | — | — | — | 0.0589 | 0.0141 | 0.0542 | 0.000 | — | — |
| Exp 1: High LR, r=8, 1 epoch | 1e-3 | 8 | 1 | 0.4327 | 0.2896 | 0.3757 | 0.120 | 2.9 min | 2.27 GB |
| Exp 2: Med LR, r=16, 2 epochs | 5e-4 | 16 | 2 | 0.4392 | 0.2929 | 0.3809 | 0.108 | 5.8 min | 3.29 GB |
| Exp 3: Low LR, r=16, 2 epochs | 3e-4 | 16 | 2 | 0.4362 | 0.2854 | 0.3759 | 0.118 | 5.8 min | 4.30 GB |
| **Exp 4: Med LR, r=32, 3 epochs ✅ Best** | **5e-4** | **32** | **3** | **0.4399** | **0.2996** | **0.3867** | **0.107** | **9.1 min** | **7.37 GB** |

Experiment 4 was selected as the best configuration and retrained on the full 1,800 examples for the final deployed model.

---

## Performance Metrics

| Metric | Base Model | Fine-Tuned | Improvement |
|---|---|---|---|
| ROUGE-1 | 0.0589 | 0.438 | +643% |
| ROUGE-2 | 0.0141 | 0.301 | +2035% |
| ROUGE-L | 0.0542 | 0.387 | **+613.5%** |
| BLEU | 0.0000 | 0.107 | — |
| Perplexity | 7.09 | 5.14 | -27.5% |

---

## Example Conversations

**Question:** What is the mechanism of action of metformin?

> **Base model:** Metformin is a drug used to treat diabetes.
>
> **Fine-tuned:** Metformin activates AMPK, reducing hepatic glucose production, improving insulin sensitivity in peripheral tissues, and decreasing intestinal glucose absorption.

---

**Question:** What are the symptoms of appendicitis?

> **Base model:** Appendicitis is a disease.
>
> **Fine-tuned:** Symptoms include periumbilical pain migrating to the right iliac fossa, nausea, vomiting, low-grade fever, and rebound tenderness at McBurney's point.

---

**Question:** How does penicillin kill bacteria?

> **Base model:** Penicillin is an antibiotic.
>
> **Fine-tuned:** Penicillin inhibits bacterial cell wall synthesis by binding to penicillin-binding proteins and preventing cross-linking of peptidoglycan chains, causing osmotic lysis and cell death.

---

## How to Run

### Option 1: Live demo, no setup needed
Open **[https://huggingface.co/spaces/Numwali/medical-chatbot-lora](https://huggingface.co/spaces/Numwali/medical-chatbot-lora)** in any browser.

---

### Option 2: Run on Google Colab

1. Click the **Open in Colab** badge at the top of this page
2. Go to **Runtime → Change runtime type → T4 GPU**
3. Run **Cell 1** (install packages)
4. Go to **Runtime → Restart session**
5. Run all remaining cells from Cell 2 onwards
6. The full pipeline including all 4 experiments takes approximately 35 minutes

---

### Option 3: Load the model directly in Python

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel
import torch

# Load tokeniser and model from Hugging Face Hub
tokenizer = AutoTokenizer.from_pretrained("Numwali/medical-chatbot-lora")
base      = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-base")
model     = PeftModel.from_pretrained(base, "Numwali/medical-chatbot-lora")
model.eval()

def ask(question):
    inputs = tokenizer(
        "Answer this medical question: " + question,
        return_tensors="pt",
        truncation=True,
        max_length=256
    )
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=128,
            num_beams=4,
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(output[0], skip_special_tokens=True)

print(ask("What is the mechanism of action of metformin?"))
```

---

## Repository Structure

```
medical-chatbot/
│
├── medical_chatbot_Assistant.ipynb    # Main notebook & complete pipeline
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
│
└── assets/                            # Charts and screenshots from the notebook
    ├── dataset_distribution.png       # Figure 1: Input and output length distributions
    ├── loss_curves.png                # Figure 2: Training and validation loss curves
    ├── rouge_comparison.png           # Figure 3: ROUGE scores across all experiments
    ├── base_vs_finetuned.png          # Figure 4: Base vs fine-tuned metrics comparison
    └── gradio_ui.png                  # Figure 5: Hugging Face Spaces screenshot
```

---

## Requirements

```
transformers==4.44.2
peft==0.12.0
accelerate==0.34.2
datasets==2.21.0
evaluate
rouge_score
sacrebleu
gradio
sentencepiece
matplotlib
pandas
torch
```

---

## Acknowledgements

- **Dataset:** [MedAlpaca](https://github.com/kbressem/medAlpaca) by Han et al., 2023
- **Fine-tuning method:** [LoRA](https://arxiv.org/abs/2106.09685) by Hu et al., 2022
- **Base model:** [Flan-T5](https://arxiv.org/abs/2210.11416) by Chung et al., 2022
- **Infrastructure:** Hugging Face Transformers, PEFT, Datasets, and Spaces
