"""
=============================================================
  Smart Energy Systems – Emotion Detection Project
  Pre-trained model: j-hartmann/emotion-english-distilroberta-base
  Emotions detected: anger, disgust, fear, joy, neutral, sadness, surprise
=============================================================

SETUP INSTRUCTIONS
------------------
1. Create a virtual environment (recommended):
       python -m venv venv
       source venv/bin/activate        # macOS/Linux
       venv\Scripts\activate           # Windows

2. Install dependencies:
       pip install -r requirements.txt

3. Run this script:
       python emotion_detector.py

   OR run the interactive demo:
       python emotion_detector.py --interactive

NOTE: On first run the model (~330 MB) is downloaded automatically
      and cached in ~/.cache/huggingface/
=============================================================
"""

import argparse
import sys

# ── 1. Check dependencies ────────────────────────────────────────────────────
try:
    from transformers import pipeline
except ImportError:
    print("[ERROR] 'transformers' not found. Run:  pip install -r requirements.txt")
    sys.exit(1)

try:
    import torch
    DEVICE = 0 if torch.cuda.is_available() else -1   # 0 = GPU, -1 = CPU
    DEVICE_NAME = "GPU" if DEVICE == 0 else "CPU"
except ImportError:
    print("[ERROR] 'torch' not found. Run:  pip install -r requirements.txt")
    sys.exit(1)


# ── 2. Load pre-trained model ────────────────────────────────────────────────
MODEL_NAME = "j-hartmann/emotion-english-distilroberta-base"

print(f"\n{'='*60}")
print("  Emotion Detection – Smart Energy Systems Project")
print(f"{'='*60}")
print(f"  Model  : {MODEL_NAME}")
print(f"  Device : {DEVICE_NAME}")
print(f"{'='*60}\n")

print("[INFO] Loading pre-trained emotion detection model...")
print("       (First run downloads ~330 MB – please wait)\n")

emotion_classifier = pipeline(
    task="text-classification",
    model=MODEL_NAME,
    top_k=None,       # return scores for ALL emotion classes
    device=DEVICE,
)
print("[INFO] Model loaded successfully!\n")


# ── 3. Helper: pretty-print results ─────────────────────────────────────────
EMOJI = {
    "anger":    "😠",
    "disgust":  "🤢",
    "fear":     "😨",
    "joy":      "😊",
    "neutral":  "😐",
    "sadness":  "😢",
    "surprise": "😲",
}

def print_results(text: str, results: list[dict]) -> None:
    """Display emotion scores as a simple bar chart in the terminal."""
    sorted_results = sorted(results, key=lambda x: x["score"], reverse=True)
    top_emotion    = sorted_results[0]["label"]
    top_score      = sorted_results[0]["score"]

    print(f"  Input   : \"{text}\"")
    print(f"  Result  : {EMOJI.get(top_emotion, '')}  {top_emotion.upper()}  "
          f"(confidence: {top_score:.1%})\n")
    print("  Score breakdown:")
    for item in sorted_results:
        label  = item["label"]
        score  = item["score"]
        bar    = "█" * int(score * 30)
        emoji  = EMOJI.get(label, " ")
        print(f"    {emoji} {label:<10} {bar:<30} {score:.1%}")
    print()


# ── 4. Analyse a batch of sample sentences ──────────────────────────────────
def run_demo() -> None:
    sample_texts = [
        # General emotions
        "I am so happy today! Everything is going perfectly.",
        "This is absolutely terrible. I can't believe this happened.",
        "I'm not sure how I feel about this situation.",
        "Oh wow, I did not see that coming at all!",
        "I'm scared about what might happen next.",
        "I feel so lonely and hopeless right now.",

        # Smart-grid / energy-related sentences (domain relevance)
        "The new smart meter installation is working flawlessly – great news!",
        "Another blackout! I'm furious about this power company.",
        "I'm worried that the grid cyberattack will affect my home.",
        "Honestly, I don't care whether they upgrade the grid or not.",
        "Renewable energy growth is incredible – what an exciting time!",
        "The electricity bill has doubled again. This is disgusting.",
    ]

    print("─" * 60)
    print("  DEMO: Analysing sample sentences")
    print("─" * 60 + "\n")

    for text in sample_texts:
        results = emotion_classifier(text)[0]   # list of {label, score}
        print_results(text, results)
        print("─" * 60 + "\n")


# ── 5. Interactive mode ──────────────────────────────────────────────────────
def run_interactive() -> None:
    print("─" * 60)
    print("  INTERACTIVE MODE  –  type your own sentences")
    print("  (type 'quit' or press Ctrl+C to exit)")
    print("─" * 60 + "\n")

    while True:
        try:
            text = input("  Enter text: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\n[INFO] Exiting. Goodbye!")
            break

        if text.lower() in {"quit", "exit", "q"}:
            print("[INFO] Exiting. Goodbye!")
            break

        if not text:
            print("  [!] Please enter some text.\n")
            continue

        results = emotion_classifier(text)[0]
        print()
        print_results(text, results)
        print("─" * 60 + "\n")


# ── 6. Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Emotion detection using a pre-trained DistilRoBERTa model."
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode (type your own sentences).",
    )
    args = parser.parse_args()

    if args.interactive:
        run_interactive()
    else:
        run_demo()
        print("[INFO] Run with --interactive to enter your own sentences.\n")
