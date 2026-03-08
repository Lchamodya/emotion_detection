# Emotion Detection – Smart Energy Systems Project
### Pre-trained NLP model | University of Vaasa – TECH1001

---

## What This Project Does

This project runs a **pre-trained emotion detection model** on text input.
It classifies any sentence into one of **7 emotions**:

| Emotion   | Emoji |
|-----------|-------|
| Joy       | 😊    |
| Anger     | 😠    |
| Sadness   | 😢    |
| Fear      | 😨    |
| Surprise  | 😲    |
| Disgust   | 🤢    |
| Neutral   | 😐    |

The model used is **`j-hartmann/emotion-english-distilroberta-base`** — a
DistilRoBERTa transformer fine-tuned on 6 emotion datasets, available
free on [Hugging Face](https://huggingface.co/j-hartmann/emotion-english-distilroberta-base).

---

## Connection to Smart Energy Systems

This project is relevant to the course in several ways:

- **Machine Learning Applications (Lecture 5):** This is a real-world
  application of a pre-trained transformer model — the same pipeline
  used for energy demand forecasting and anomaly detection.

- **Data-driven AI approach:** Just like load forecasting (collect data →
  define model → train → predict), emotion detection follows the same
  paradigm taught by Prof. Elmusrati.

- **Human-in-the-loop (Lecture 10):** Emotion detection can be used in
  smart grid customer service systems to route frustrated consumers to
  human operators automatically.

- **Energy security & public trust:** Understanding public sentiment
  about grid outages, renewable transitions, or pricing policies is
  increasingly important for utilities.

---

## Project Structure

```
emotion_detection/
├── emotion_detector.py     ← main script
├── requirements.txt        ← all dependencies
└── README.md               ← this file
```

---

## Setup Instructions

### Step 1 – Make sure Python is installed
```bash
python --version    # should be 3.9 or higher
```

### Step 2 – Create a virtual environment (recommended)
```bash
# Create
python -m venv venv

# Activate – macOS / Linux
source venv/bin/activate

# Activate – Windows
venv\Scripts\activate
```

### Step 3 – Install dependencies
```bash
pip install -r requirements.txt
```
This installs PyTorch, Hugging Face Transformers, and utilities.
> **Note:** First install may take 2–5 minutes depending on your connection.

### Step 4 – Run the demo
```bash
python emotion_detector.py
```
On **first run**, the model (~330 MB) is downloaded automatically and
cached in `~/.cache/huggingface/`. Subsequent runs are instant.

### Step 5 – Interactive mode (type your own sentences)
```bash
python emotion_detector.py --interactive
```

---

## Example Output

```
============================================================
  Emotion Detection – Smart Energy Systems Project
============================================================
  Model  : j-hartmann/emotion-english-distilroberta-base
  Device : CPU
============================================================

  Input   : "Another blackout! I'm furious about this power company."
  Result  : 😠  ANGER  (confidence: 91.3%)

  Score breakdown:
    😠 anger      ██████████████████████████     91.3%
    😢 sadness    ███                             4.1%
    😐 neutral    █                               2.0%
    😨 fear       █                               1.5%
    😊 joy                                        0.7%
    🤢 disgust                                    0.3%
    😲 surprise                                   0.1%
```

---

## How It Works (Technical Explanation)

```
Your text
    │
    ▼
Tokenizer
(splits text into tokens the model understands)
    │
    ▼
DistilRoBERTa encoder
(12-layer transformer, 82M parameters)
(pre-trained on massive text corpus)
(fine-tuned on emotion datasets)
    │
    ▼
Classification head
(7-class softmax output)
    │
    ▼
Emotion scores (sum to 100%)
```

**DistilRoBERTa** is a smaller, faster version of RoBERTa (itself an
improved BERT). It was fine-tuned on 6 emotion datasets:
- CrowdFlower
- Emotion Dataset (Elvis)
- GoEmotions
- ISEAR
- MELD
- SemEval-2018

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: torch` | Run `pip install -r requirements.txt` |
| Slow first run | Normal — model downloading (~330 MB) |
| `No space left on device` | Free ~500 MB disk space |
| Model gives wrong results | Try longer, clearer sentences |
| CUDA/GPU errors | The script auto-falls back to CPU |

---

## Next Steps / Extensions

1. **Batch processing from CSV** — read a file of customer complaints and
   classify all of them automatically
2. **Real-time dashboard** — combine with `streamlit` to build a web UI
3. **Fine-tuning** — retrain the model on energy-domain specific text
4. **Multi-language** — swap to `xlm-roberta` for non-English text
5. **Integration with smart grid data** — analyse social media posts
   about grid outages for situational awareness

---

*Project for TECH1001 – Smart Energy Systems, University of Vaasa, 2026*
