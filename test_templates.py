#!/usr/bin/env python3
"""
Test script for the template system.
"""

import sys
sys.path.append('.')

from templates.template_model import TemplateManager

def test_template_loading():
    """Test loading templates from JSON files."""
    print("Testing template loading...")
    tm = TemplateManager()
    
    print(f"✅ Loaded {len(tm.templates)} templates")
    
    for template in tm.templates:
        print(f"   - {template.vendor_name} (ID: {template.vendor_id})")
    
    return tm

def test_template_matching(tm):
    """Test template matching with sample text."""
    print("\nTesting template matching...")
    
    # Test cases
    test_cases = [
        ("WALMART SUPERCENTER\nSave Money. Live Better", "walmart"),
        ("TARGET\nExpect More. Pay Less", "target"),
        ("COSTCO WHOLESALE\nWarehouse #123", "costco"),
        ("Kroger\n123 Main Street", "kroger"),
        ("Random Store\nNo template match", None)
    ]
    
    for text, expected in test_cases:
        matched = tm.find_matching_template(text)
        if matched:
            print(f"✅ '{text.split()[0]}' matched template: {matched.vendor_name}")
        else:
            print(f"❌ '{text.split()[0]}' no template match (expected: {expected})")

def test_data_extraction(tm):
    """Test data extraction with a sample receipt."""
    print("\nTesting data extraction...")
    
    sample_receipt = """
WALMART SUPERCENTER
Store #1234
123 Main Street, Anytown, ST 12345
(555) 123-4567

MILK WHOLE GAL                  $3.98
BREAD WHITE LOAF                $2.50
BANANA LB                       $1.20

SUBTOTAL                        $7.68
SALES TAX                       $0.55
TOTAL                           $8.23

VISA CREDIT                     $8.23
TC#: 1234567890123

Thank you for shopping!
"""
    
    result = tm.parse_receipt(sample_receipt)
    
    if result.get('template_match'):
        print(f"✅ Template matched: {result.get('vendor')}")
        print(f"   Store: {result.get('store_name', 'N/A')}")
        print(f"   Items found: {len(result.get('items', []))}")
        if result.get('items'):
            for item in result.get('items', []):
                print(f"     - {item['name']}: {item['price']}")
        print(f"   Subtotal: {result.get('subtotal', 'N/A')}")
        print(f"   Tax: {result.get('tax', 'N/A')}")
        print(f"   Total: {result.get('total', 'N/A')}")
    else:
        print("❌ No template match for sample receipt")

def test_target_receipt(tm):
    """Test Target receipt parsing."""
    print("\nTesting Target receipt...")
    
    target_receipt = """
TARGET
Expect More. Pay Less.
Store T-1234
555 Market St, City, ST 12345

BANANAS ORGANIC                 $2.49
MILK 2% GALLON                  $3.29
BREAD WHEAT                     $2.99

SUBTOTAL                        $8.77
SALES TAX                       $0.62
TOTAL                           $9.39

REDCARD DEBIT                   $9.39
REF# 1234-5678-9012

Thank you for shopping at Target!
"""
    
    result = tm.parse_receipt(target_receipt)
    
    if result.get('template_match'):
        print(f"✅ Template matched: {result.get('vendor')}")
        print(f"   Items found: {len(result.get('items', []))}")
        if result.get('items'):
            for item in result.get('items', []):
                print(f"     - {item['name']}: {item['price']}")
        print(f"   Total: {result.get('total', 'N/A')}")
    else:
        print("❌ No template match for Target receipt")

if __name__ == "__main__":
    try:
        tm = test_template_loading()
        test_template_matching(tm)
        test_data_extraction(tm)
        test_target_receipt(tm)
        print("\n🎉 All tests completed!")
    except Exception as e:
        print(f"❌ Error during testing: {e}")
        import traceback
        traceback.print_exc() 