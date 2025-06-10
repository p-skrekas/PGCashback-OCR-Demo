import re

def normalize_text(text):
    """Normalize text for comparison with better receipt-specific handling"""
    if not text:
        return ""
    
    # Convert to lowercase and strip
    text = text.lower().strip()
    
    # Normalize date formats: convert various date separators to consistent format
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1\2\3', text)
    
    # Normalize currency and amounts
    text = re.sub(r'rm\s*', 'rm', text)  # "RM 82.80" -> "rm82.80"
    text = re.sub(r'\$\s*', 'usd', text)  # "$25.50" -> "usd25.50"
    
    # Remove common receipt labels that might differ
    text = re.sub(r'\b(total\s*sales?|sales?\s*total|total)\s*(inclusive\s*)?(\(.*?\))?\s*(rm|usd|\$)?\s*', 'total ', text)
    text = re.sub(r'\b(date|dt)\s*:?\s*', '', text)
    
    # Multiple spaces to single space
    text = re.sub(r'\s+', ' ', text)
    
    # Remove most punctuation but keep currency decimals
    text = re.sub(r'[^\w\s\.]', '', text)
    
    return text.strip()

def filter_hypothesis_text(reference_text, hypothesis_text):
    """Filter hypothesis text to only include words that exist in the reference text"""
    if not reference_text or not hypothesis_text:
        return ""
    
    # Normalize and tokenize
    ref_words = set(normalize_text(reference_text).split())
    hyp_words = normalize_text(hypothesis_text).split()
    
    # Keep only hypothesis words that exist in reference
    filtered_words = [word for word in hyp_words if word in ref_words]
    
    return ' '.join(filtered_words)

def calculate_simple_wer(reference_text, hypothesis_text):
    """Simple WER calculation"""
    if not reference_text or not hypothesis_text:
        return 1.0
    
    # Filter hypothesis to only include words from reference
    filtered_hypothesis = filter_hypothesis_text(reference_text, hypothesis_text)
    
    # Normalize and tokenize
    ref_words = normalize_text(reference_text).split()
    hyp_words = normalize_text(filtered_hypothesis).split()
    
    # Simple WER calculation
    ref_count = len(ref_words)
    hyp_count = len(hyp_words)
    
    # Count matching words
    ref_set = set(ref_words)
    hyp_set = set(hyp_words)
    matching_words = len(ref_set.intersection(hyp_set))
    
    # Simple WER approximation
    wer = (ref_count - matching_words) / ref_count if ref_count > 0 else 1.0
    
    return min(wer, 1.0)  # Cap at 100%

# Test with your sample data
def test_normalization():
    print("🧪 TESTING IMPROVED TEXT NORMALIZATION")
    print("="*60)
    
    # Your actual ground truth
    ground_truth = "03-02-16 TOTAL SALES (INCLUSIVE GST) RM 82.80"
    
    # Simulated OCR outputs (what might actually be extracted)
    test_cases = [
        "Date: 03/02/16 Total: RM 82.80",  # Typical OCR output
        "03/02/2016 Total Sales RM82.80",   # Different format
        "Date 03-02-16 TOTAL RM 82.80",     # Mixed format
        "03 02 16 Total RM 82.80",          # Spaced date
        "Total: RM82.80 Date: 03/02/16",    # Different order
    ]
    
    print(f"📋 Ground Truth: '{ground_truth}'")
    print(f"📋 Normalized GT: '{normalize_text(ground_truth)}'")
    print(f"📋 GT Words: {normalize_text(ground_truth).split()}")
    print()
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"🧪 Test Case {i}: '{test_case}'")
        normalized = normalize_text(test_case)
        filtered = filter_hypothesis_text(ground_truth, test_case)
        wer = calculate_simple_wer(ground_truth, test_case)
        
        print(f"   📝 Normalized: '{normalized}'")
        print(f"   🎯 Filtered: '{filtered}'")
        print(f"   📊 WER: {wer:.1%}")
        print()

if __name__ == "__main__":
    test_normalization() 