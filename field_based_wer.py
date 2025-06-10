import json
import re
from pathlib import Path

def normalize_text(text):
    """Normalize text for comparison with better receipt-specific handling"""
    if not text:
        return ""
    
    text = text.lower().strip()
    
    # Normalize date formats: convert various date separators to consistent format
    text = re.sub(r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})', r'\1\2\3', text)
    
    # Normalize currency
    text = re.sub(r'rm\s*', 'rm', text)
    text = re.sub(r'\$\s*', 'usd', text)
    
    # Remove common receipt labels
    text = re.sub(r'\b(total\s*sales?|sales?\s*total|total)\s*(inclusive\s*)?(\(.*?\))?\s*(rm|usd|\$)?\s*', 'total ', text)
    text = re.sub(r'\b(date|dt)\s*:?\s*', '', text)
    
    # Clean up spacing and punctuation
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.]', '', text)
    
    return text.strip()

def extract_field_values_from_gt(ground_truth_data, receipt_id):
    """Extract individual field values from ground truth JSON"""
    if receipt_id not in ground_truth_data:
        return {}
    
    gt_entry = ground_truth_data[receipt_id]
    if 'raw_data' not in gt_entry:
        return {}
    
    fields = {}
    raw_data = gt_entry['raw_data']
    
    if 'important_words' in raw_data:
        for item in raw_data['important_words']:
            label = item.get('label', '')
            if 'words' in item:
                # Combine all text for this field
                field_text = ' '.join([word.get('Text', '') for word in item['words']])
                fields[label] = normalize_text(field_text)
    
    return fields

def extract_field_values_from_text(extracted_text):
    """Extract field values from OCR/LLM extracted text"""
    if not extracted_text:
        return {}
    
    text = extracted_text.lower()
    fields = {}
    
    # Extract date patterns
    date_patterns = [
        r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',  # DD/MM/YYYY or DD-MM-YY
        r'(\d{4}[/-]\d{1,2}[/-]\d{1,2})',    # YYYY/MM/DD
    ]
    
    for pattern in date_patterns:
        match = re.search(pattern, text)
        if match:
            fields['date'] = normalize_text(match.group(1))
            break
    
    # Extract total/amount patterns
    amount_patterns = [
        r'(total[^0-9]*(?:rm|usd|\$)?\s*\d+\.?\d*)',  # total ... amount
        r'((?:rm|usd|\$)\s*\d+\.?\d*)',               # currency amount
        r'(\d+\.\d{2})',                              # decimal amount
    ]
    
    for pattern in amount_patterns:
        match = re.search(pattern, text)
        if match:
            fields['total'] = normalize_text(match.group(1))
            break
    
    # Extract company/store name (first line or capitalized words)
    lines = extracted_text.split('\n')
    if lines:
        potential_company = lines[0].strip()
        if len(potential_company) > 3:  # Reasonable company name length
            fields['company'] = normalize_text(potential_company)
    
    return fields

def calculate_field_based_wer(gt_fields, extracted_fields):
    """Calculate WER based on individual fields rather than concatenated text"""
    if not gt_fields or not extracted_fields:
        return 1.0, {}
    
    field_wers = {}
    total_weight = 0
    weighted_wer_sum = 0
    
    # Field weights (importance)
    weights = {
        'date': 1.0,
        'total': 2.0,  # Total is more important
        'company': 1.5,
        'address': 1.0
    }
    
    for field, gt_value in gt_fields.items():
        if not gt_value:
            continue
            
        weight = weights.get(field, 1.0)
        total_weight += weight
        
        if field in extracted_fields and extracted_fields[field]:
            # Compare field values
            gt_words = set(gt_value.split())
            ext_words = set(extracted_fields[field].split())
            
            if gt_words and ext_words:
                # Calculate overlap
                common_words = len(gt_words.intersection(ext_words))
                field_wer = max(0, (len(gt_words) - common_words) / len(gt_words))
            else:
                field_wer = 1.0
        else:
            # Field not found
            field_wer = 1.0
        
        field_wers[field] = field_wer
        weighted_wer_sum += field_wer * weight
    
    # Calculate overall weighted WER
    overall_wer = weighted_wer_sum / total_weight if total_weight > 0 else 1.0
    
    return min(overall_wer, 1.0), field_wers

def test_field_based_approach():
    """Test the field-based WER approach"""
    print("🧪 TESTING FIELD-BASED WER APPROACH")
    print("="*60)
    
    # Simulate ground truth fields (from JSON)
    gt_fields = {
        'date': normalize_text('03-02-16'),
        'total': normalize_text('TOTAL SALES (INCLUSIVE GST) RM 82.80'),
        'company': normalize_text('Some Store Name')
    }
    
    print("📋 Ground Truth Fields:")
    for field, value in gt_fields.items():
        print(f"  {field}: '{value}'")
    print()
    
    # Test different OCR/LLM extraction scenarios
    test_cases = [
        {
            'name': 'Good OCR Match',
            'text': 'SOME STORE NAME\nDate: 03/02/16\nTotal: RM 82.80'
        },
        {
            'name': 'Partial OCR Match',
            'text': 'STORE NAME\n03-02-2016\nTotal RM82.80'
        },
        {
            'name': 'Poor OCR Match',
            'text': 'Some text\nDate unclear\nAmount: 80.00'
        },
        {
            'name': 'Different Format',
            'text': 'Company Store\n16/02/03\nRM 82.80 Total'
        }
    ]
    
    for test_case in test_cases:
        print(f"🧪 {test_case['name']}: '{test_case['text']}'")
        
        # Extract fields from the test text
        extracted_fields = extract_field_values_from_text(test_case['text'])
        print(f"   📝 Extracted fields: {extracted_fields}")
        
        # Calculate field-based WER
        overall_wer, field_wers = calculate_field_based_wer(gt_fields, extracted_fields)
        
        print(f"   📊 Overall WER: {overall_wer:.1%}")
        print(f"   📊 Field WERs: {field_wers}")
        print()

if __name__ == "__main__":
    test_field_based_approach() 