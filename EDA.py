import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import random
from pathlib import Path
import sys
import time
import json
import pandas as pd
import seaborn as sns
from datetime import datetime
import warnings
import re
import difflib
from collections import Counter
import base64

warnings.filterwarnings('ignore')

# Import functions from app.py
import importlib.util
spec = importlib.util.spec_from_file_location("app", "app.py")
app_module = importlib.util.module_from_spec(spec)
sys.modules["app"] = app_module
spec.loader.exec_module(app_module)

from app import (
    setup_google_vision, 
    setup_vertex_ai, 
    extract_text_with_vision, 
    analyze_text_with_llm, 
    analyze_receipt_with_llm
)

# Initialize clients
vision_client = setup_google_vision()
vertex_model = setup_vertex_ai()

def calculate_word_error_rate(reference_text, hypothesis_text):
    if not reference_text or not hypothesis_text:
        return 1.0
    ref_words = reference_text.split()
    hyp_words = hypothesis_text.split()
    r = len(ref_words)
    h = len(hyp_words)
    dp = np.zeros((r + 1, h + 1))
    for i in range(r + 1):
        dp[i][0] = i
    for j in range(h + 1):
        dp[0][j] = j
    for i in range(1, r + 1):
        for j in range(1, h + 1):
            if ref_words[i-1] == hyp_words[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = 1 + min(
                    dp[i-1][j],
                    dp[i][j-1],
                    dp[i-1][j-1]
                )
    i, j = r, h
    substitutions = 0
    deletions = 0
    insertions = 0
    while i > 0 or j > 0:
        if i > 0 and j > 0 and ref_words[i-1] == hyp_words[j-1]:
            i -= 1
            j -= 1
        elif i > 0 and j > 0 and dp[i][j] == dp[i-1][j-1] + 1:
            substitutions += 1
            i -= 1
            j -= 1
        elif i > 0 and dp[i][j] == dp[i-1][j] + 1:
            deletions += 1
            i -= 1
        elif j > 0 and dp[i][j] == dp[i][j-1] + 1:
            insertions += 1
            j -= 1
        else:
            break
    total_errors = substitutions + deletions + insertions
    wer = total_errors / r if r > 0 else 1.0
    wer = min(wer, 1.0)
    return wer

def extract_full_text_with_llm(image_bytes, vertex_model):
    """Extract the complete text that the LLM sees from an image"""
    try:
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = "image/jpeg"
        from vertexai.generative_models import Part
        image_part = Part.from_data(
            data=base64.b64decode(b64_image),
            mime_type=mime_type
        )
        prompt = (
            "Please extract ALL the text you can see in this receipt image. "
            "Return ONLY the raw text as it appears, line by line, without any formatting or analysis. "
            "Do not add any JSON structure or explanations - just the pure text content. "
            "If you see text that is unclear or partially visible, include your best interpretation. "
            "Preserve the approximate line structure as you see it."
        )
        response = vertex_model.generate_content([prompt, image_part])
        return response.text.strip() if response.text else ""
    except Exception as e:
        return ""

def standardize_text_lines(text):
    """Standardize text to have one line per logical line, strip spaces, and collapse multiple newlines."""
    if not text:
        return ""
    # Split, strip each line, remove empty lines, and join with single newlines
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return '\n'.join(lines)

def run_wer_on_new_dataset():
    """
    Compute WER for the two images in new_dataset using the ground truth in new_dataset/data.json.
    Compares both OCR+LLM and Direct LLM approaches.
    """
    import json
    from pathlib import Path

    dataset_dir = Path("new_dataset")
    images_dir = dataset_dir / "images"
    gt_path = dataset_dir / "data.json"

    # Load ground truth
    with open(gt_path, "r") as f:
        gt_data = json.load(f)["data"]

    # Map image filename to ground truth
    gt_map = {}
    for entry in gt_data:
        img_name = entry["image"]
        if not (images_dir / Path(img_name).name).exists():
            if not img_name.endswith(".jpg") and (images_dir / (img_name + ".jpg")).exists():
                img_name = img_name + ".jpg"
            else:
                alt_name = Path(img_name).stem + ".jpg"
                if (images_dir / alt_name).exists():
                    img_name = alt_name
        gt_map[Path(img_name).name] = entry["ground_truth"]

    results = []
    for img_name, gt_text in gt_map.items():
        img_path = images_dir / img_name
        print(f"\nProcessing {img_name}")
        # Standardize all text for fair comparison
        gt_text_std = standardize_text_lines(gt_text)
        print("  Ground truth length:", len(gt_text_std.split()))
        if not img_path.exists():
            print(f"  Image not found: {img_path}")
            continue
        with open(img_path, "rb") as f:
            image_bytes = f.read()

        # OCR+LLM pipeline
        ocr_text = None
        ocr_wer_result = None
        ocr_text_std = None
        try:
            extracted_text, _ = extract_text_with_vision(image_bytes, vision_client)
            if extracted_text:
                ocr_text_std = standardize_text_lines(extracted_text)
                print("  --- OCR+LLM Output ---")
                print(ocr_text_std)
                ocr_wer_result = calculate_word_error_rate(gt_text_std, ocr_text_std)
                print(f"  OCR+LLM WER: {ocr_wer_result:.1%}")
            else:
                print("  OCR+LLM failed: No text extracted")
        except Exception as e:
            print(f"  OCR+LLM failed: {e}")

        # Direct LLM pipeline
        direct_llm_text = None
        direct_wer_result = None
        direct_llm_text_std = None
        try:
            direct_llm_text = extract_full_text_with_llm(image_bytes, vertex_model)
            if direct_llm_text:
                direct_llm_text_std = standardize_text_lines(direct_llm_text)
                print("  --- Direct LLM Output ---")
                print(direct_llm_text_std)
                direct_wer_result = calculate_word_error_rate(gt_text_std, direct_llm_text_std)
                print(f"  Direct LLM WER: {direct_wer_result:.1%}")
            else:
                print("  Direct LLM failed: No text extracted")
        except Exception as e:
            print(f"  Direct LLM failed: {e}")

        results.append({
            "image": img_name,
            "ocr_wer": ocr_wer_result,
            "direct_wer": direct_wer_result
        })

    # Print summary
    ocr_wers = [r["ocr_wer"] for r in results if r["ocr_wer"] is not None]
    direct_wers = [r["direct_wer"] for r in results if r["direct_wer"] is not None]
    if ocr_wers:
        print(f"\nAverage OCR+LLM WER: {sum(ocr_wers)/len(ocr_wers):.1%}")
    if direct_wers:
        print(f"Average Direct LLM WER: {sum(direct_wers)/len(direct_wers):.1%}")
    if ocr_wers and direct_wers:
        if sum(ocr_wers)/len(ocr_wers) < sum(direct_wers)/len(direct_wers):
            print("🏆 OCR+LLM performs better on new_dataset")
        elif sum(direct_wers)/len(direct_wers) < sum(ocr_wers)/len(ocr_wers):
            print("🏆 Direct LLM performs better on new_dataset")
        else:
            print("🤝 Both methods have similar WER performance on new_dataset")

if __name__ == "__main__":
    print("🎯 WORD ERROR RATE (WER) ANALYSIS - new_dataset only")
    run_wer_on_new_dataset()
