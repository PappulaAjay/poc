## Healthcare SLM Fine-Tuning POC (CPU LoRA)

This repo is a working POC for:
- **Upload healthcare data** (PDF / CSV / TXT, ~100MB)
- **CPU-only LoRA fine-tuning** (small SLM by default; finishes in minutes)
- **Training metrics dashboard (VERY IMPORTANT)**: live loss, perplexity, token-accuracy (approx)
- **Chatbot demo**: Before vs After training

### Quickstart

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt
python3 -m streamlit run app.py
```

### Best “WOW” dataset format

Upload a CSV with columns:
- `question`
- `answer`

This produces the strongest before/after demo because the model learns direct healthcare Q&A pairs.

### Notes

- **This is fine-tuning (LoRA)**: the adapter is saved to `runs/<timestamp>/lora_adapter`.
- CPU LoRA on very large models may be slow; the default base model is small to reliably complete quickly.