#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Batch label `document_content` into text-type categories using a local HF model (GPU).

Key features
- Numeric-code protocol (1–5) for robust, compact outputs
- TSV-style parsing (accepts tab/colon/hyphen separators)
- Batched generation with retries + strict reminders
- Per-item strict fallback (never stalls if a batch is messy)
- Incremental saving to out_csv (resume-friendly)
- Fast defaults for RTX A6000 (FP16, truncation)

Example:
python label_with_llama.py \
  --in_csv merged_similarity.csv \
  --out_csv merged_labeled.csv \
  --text_col document_content \
  --out_col text_type_category \
  --batch_size 48 \
  --max_chars 1000 \
  --max_new_tokens 96 \
  --temperature 0.0 \
  --model_name meta-llama/Meta-Llama-3.1-8B-Instruct \
  --resume
"""

import argparse
import os
import re
import sys
import time
from typing import Dict, List

import pandas as pd
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ---- Categories & codes ----
CATEGORIES = [
    "Dilemma / Case Example",        # code 1
    "Rights / Legal / Policy",       # code 2
    "Educational / Explanatory",     # code 3
    "Research / Scholarly Analysis", # code 4
    "Other / Miscellaneous",         # code 5
]

CODE_TO_LABEL = {
    "1": CATEGORIES[0],
    "2": CATEGORIES[1],
    "3": CATEGORIES[2],
    "4": CATEGORIES[3],
    "5": CATEGORIES[4],
}

# ---- Prompt builder (numeric codes, TSV-like) ----
def make_prompt(batch_items: List[Dict], max_chars: int) -> str:
    """
    Require output EXACTLY one line per item: <id>\t<code>
    where code ∈ {1,2,3,4,5}.
    """
    header = (
        "You are a labeling function. For each text, output EXACTLY one line: <id>\\t<code>\n"
        "Codes:\n"
        "1 = Dilemma / Case Example\n"
        "2 = Rights / Legal / Policy\n"
        "3 = Educational / Explanatory\n"
        "4 = Research / Scholarly Analysis\n"
        "5 = Other / Miscellaneous\n\n"
        "STRICT OUTPUT RULES:\n"
        "- One line per id ONLY, no extra text before or after.\n"
        "- Format each line EXACTLY as: <id>\\t<code>\n"
        "- No JSON. No bullets. No commentary.\n\n"
        "TEXTS:\n"
    )
    lines = []
    for it in batch_items:
        txt = (it["text"] or "").strip()
        if len(txt) > max_chars:
            txt = txt[:max_chars]
        safe = txt.replace('"""', '\\"""')
        lines.append(f'{it["id"]}: """{safe}"""')
    return header + "\n".join(lines) + "\n\nRESULTS:"

# ---- Parser (accepts tab OR colon OR hyphen) ----
_CODE_LINE = re.compile(r"^\s*(\d+)\s*[\t:\-]\s*([1-5])\s*$")

def parse_code_lines(text: str) -> Dict[int, str]:
    """
    Accept lines like:
      123\t3
      123: 3
      123 - 3
    Return {id: label_string}
    """
    out: Dict[int, str] = {}
    for line in text.splitlines():
        m = _CODE_LINE.match(line)
        if m:
            idx, code = int(m.group(1)), m.group(2)
            out[idx] = CODE_TO_LABEL.get(code, CATEGORIES[4])
    if not out:
        raise ValueError("No <id>\\t<code> lines found.")
    return out

# ---- Text generation ----
def generate_text(model, tokenizer, prompt: str, max_new_tokens: int, temperature: float, device: str) -> str:
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out_ids = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False if temperature == 0 else True,
        temperature=(temperature if temperature > 0 else 1.0),
        top_p=1.0,
        repetition_penalty=1.0,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()

# ---- Model loader (tries FlashAttention2; falls back cleanly) ----
def load_model_and_tokenizer(model_name: str):
    print(f"Loading model: {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",  # if available
        )
    except Exception:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.float16,
        )
    return model, tok

# ---- Strict single-item fallback (SAFE) ----
def classify_single_strict(model, tok, device, text: str) -> str:
    """
    Last-chance labeling for a single item; returns a label string.
    Never raises if the model drifts; defaults to 'Other / Miscellaneous'.
    """
    p = (
        "Return exactly one line: <code>\n"
        "Codes: 1=Dilemma, 2=Rights, 3=Educational, 4=Research, 5=Other\n\n"
        f'TEXT:\n"""{(text or "")[:800]}"""\n\nRESULT:\n'
    )
    raw = generate_text(
        model=model,
        tokenizer=tok,
        prompt=p,
        max_new_tokens=8,
        temperature=0.0,
        device=device,
    )
    m = re.search(r"\b([1-5])\b", raw)
    if not m:
        return CATEGORIES[4]  # default to Other/Misc
    return CODE_TO_LABEL.get(m.group(1), CATEGORIES[4])

# ---- Main ----
def main():
    ap = argparse.ArgumentParser(description="Batch label texts into categories using a local HF model (numeric TSV protocol).")
    ap.add_argument("--in_csv", required=True, help="Input CSV path")
    ap.add_argument("--out_csv", required=True, help="Output CSV path (incrementally overwritten)")
    ap.add_argument("--text_col", default="document_content", help="Column containing text to classify")
    ap.add_argument("--out_col", default="text_type_category", help="Output label column")
    ap.add_argument("--batch_size", type=int, default=48, help="Rows per batch (48 good for A6000)")
    ap.add_argument("--max_chars", type=int, default=1000, help="Truncate long docs; classification doesn’t need full text")
    ap.add_argument("--model_name", default="meta-llama/Meta-Llama-3.1-8B-Instruct", help="HF model repo id")
    ap.add_argument("--temperature", type=float, default=0.0, help="0 = deterministic")
    ap.add_argument("--max_new_tokens", type=int, default=96, help="Generation cap for code lines")
    ap.add_argument("--resume", action="store_true", help="If out_csv exists, keep existing labels and only fill missing")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("⚠️ CUDA not detected—this script is intended for a GPU server. It will still run but will be slow.", file=sys.stderr)

    # Load data
    df = pd.read_csv(args.in_csv)
    if args.text_col not in df.columns:
        raise ValueError(f"Text column '{args.text_col}' not found in CSV.")

    if args.out_col not in df.columns:
        df[args.out_col] = pd.NA

    # Resume-aware: if --resume and out_csv exists, prefer already-labeled rows from that file
    if args.resume and os.path.exists(args.out_csv):
        try:
            existing = pd.read_csv(args.out_csv)
            if args.out_col in existing.columns and len(existing) == len(df):
                df[args.out_col] = existing[args.out_col]
                print(f"Resuming from existing labels in {args.out_csv}")
        except Exception:
            pass

    unlabeled_idx = [int(i) for i, v in enumerate(df[args.out_col].tolist()) if pd.isna(v)]

    # Load model
    model, tok = load_model_and_tokenizer(args.model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using model: {args.model_name}")
    print(f"Total rows: {len(df)} | Already labeled: {len(df) - len(unlabeled_idx)} | Remaining: {len(unlabeled_idx)}")

    # Process in batches
    start_time = time.perf_counter()
    for s in tqdm(range(0, len(unlabeled_idx), args.batch_size), desc="Batching"):
        batch_idx = unlabeled_idx[s : s + args.batch_size]
        batch_items = [{"id": int(i), "text": str(df.at[i, args.text_col])} for i in batch_idx]

        prompt = make_prompt(batch_items, args.max_chars)

        # Try 3 times with increasingly strict reminder
        result_map: Dict[int, str] = {}
        last_err = None
        for attempt in range(3):
            try:
                raw = generate_text(
                    model=model,
                    tokenizer=tok,
                    prompt=prompt,
                    max_new_tokens=args.max_new_tokens,
                    temperature=args.temperature,
                    device=device,
                )
                result_map = parse_code_lines(raw)
                break
            except Exception as e:
                last_err = e
                prompt = (
                    prompt
                    + "\n\nREMINDER: Output ONLY lines in the exact format <id>\\t<code> "
                      "where code ∈ {1,2,3,4,5}. No extra text."
                )
                time.sleep(2 ** attempt)

        # Fill labels; any missing id -> strict per-item fallback
        for i in batch_idx:
            if i in result_map:
                df.at[i, args.out_col] = result_map[i]
            else:
                df.at[i, args.out_col] = classify_single_strict(
                    model, tok, device, str(df.at[i, args.text_col])
                )

        # Incremental save (resume-friendly)
        df.to_csv(args.out_csv, index=False)

    elapsed = time.perf_counter() - start_time
    print(f"Labeled column '{args.out_col}' written to: {args.out_csv}  | Elapsed: {elapsed/60:.1f} min")

if __name__ == "__main__":
    main()
