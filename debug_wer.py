import json
from pathlib import Path

# Quick debug to see what we're comparing
def debug_text_comparison():
    print("🔍 DEBUGGING TEXT COMPARISON")
    print("="*50)
    
    # Load a sample ground truth file
    gt_dir = Path("fullDataset/info_data")
    sample_file = list(gt_dir.glob("*.json"))[0]
    
    print(f"📄 Sample file: {sample_file.name}")
    
    with open(sample_file, 'r') as f:
        data = json.load(f)
    
    # Extract text the same way as our WER code
    all_text = []
    if 'important_words' in data:
        for item in data['important_words']:
            if 'words' in item:
                for word in item['words']:
                    if 'Text' in word:
                        all_text.append(word['Text'])
    
    ground_truth_text = ' '.join(all_text)
    
    print(f"\n📋 Ground Truth Text ({len(ground_truth_text.split())} words):")
    print(f"'{ground_truth_text}'")
    
    print(f"\n🔍 Sample words: {ground_truth_text.split()[:10]}")
    
    # Show structure
    print(f"\n📊 JSON Structure:")
    if 'important_words' in data:
        for item in data['important_words']:
            print(f"  Label: {item.get('label', 'unknown')}")
            if 'words' in item:
                for word in item['words']:
                    print(f"    Text: '{word.get('Text', 'N/A')}'")

if __name__ == "__main__":
    debug_text_comparison() 