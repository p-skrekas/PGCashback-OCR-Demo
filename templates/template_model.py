"""
Template model for vendor-specific receipt parsing.
This module defines the structure for receipt templates and provides functionality
for template matching and application.
"""

import re
import json
import os
from typing import Dict, List, Pattern, Tuple, Optional, Any


class ReceiptTemplate:
    """Template for a specific vendor's receipt format."""
    
    def __init__(
        self,
        vendor_id: str,
        vendor_name: str,
        recognition_patterns: List[str],
        header_patterns: Dict[str, str] = None,
        item_pattern: str = None,
        total_pattern: str = None,
        subtotal_pattern: str = None,
        tax_pattern: str = None,
        date_pattern: str = None,
        time_pattern: str = None,
        receipt_number_pattern: str = None,
        payment_method_pattern: str = None,
        address_pattern: str = None,
        phone_pattern: str = None,
    ):
        """
        Initialize a receipt template.
        
        Args:
            vendor_id: Unique identifier for the vendor
            vendor_name: Human-readable name of the vendor
            recognition_patterns: List of regex patterns to recognize this vendor's receipts
            header_patterns: Patterns to extract header information
            item_pattern: Pattern to extract individual items
            total_pattern: Pattern to extract the total amount
            subtotal_pattern: Pattern to extract the subtotal
            tax_pattern: Pattern to extract tax information
            date_pattern: Pattern to extract the date
            time_pattern: Pattern to extract the time
            receipt_number_pattern: Pattern to extract receipt number
            payment_method_pattern: Pattern to extract payment method
            address_pattern: Pattern to extract store address
            phone_pattern: Pattern to extract phone number
        """
        self.vendor_id = vendor_id
        self.vendor_name = vendor_name
        self.recognition_patterns = [re.compile(p, re.IGNORECASE) for p in recognition_patterns]
        
        # Default patterns
        self.header_patterns = {}
        if header_patterns:
            self.header_patterns = {k: re.compile(v, re.IGNORECASE) for k, v in header_patterns.items()}
        
        self.item_pattern = re.compile(item_pattern, re.IGNORECASE) if item_pattern else None
        self.total_pattern = re.compile(total_pattern, re.IGNORECASE) if total_pattern else None
        self.subtotal_pattern = re.compile(subtotal_pattern, re.IGNORECASE) if subtotal_pattern else None
        self.tax_pattern = re.compile(tax_pattern, re.IGNORECASE) if tax_pattern else None
        self.date_pattern = re.compile(date_pattern, re.IGNORECASE) if date_pattern else None
        self.time_pattern = re.compile(time_pattern, re.IGNORECASE) if time_pattern else None
        self.receipt_number_pattern = re.compile(receipt_number_pattern, re.IGNORECASE) if receipt_number_pattern else None
        self.payment_method_pattern = re.compile(payment_method_pattern, re.IGNORECASE) if payment_method_pattern else None
        self.address_pattern = re.compile(address_pattern, re.IGNORECASE) if address_pattern else None
        self.phone_pattern = re.compile(phone_pattern, re.IGNORECASE) if phone_pattern else None
        
    def matches(self, text: str) -> bool:
        """Check if the given text matches this template."""
        for pattern in self.recognition_patterns:
            if pattern.search(text):
                return True
        return False
    
    def is_excluded_line(self, line: str) -> bool:
        """Check if a line should be excluded from item parsing."""
        line_lower = line.lower().strip()
        
        # Exclude common non-item patterns
        exclusion_patterns = [
            r'(?:sub)?total',
            r'tax',
            r'balance',
            r'amount\s+(?:due|paid)',
            r'change',
            r'cash',
            r'credit',
            r'debit',
            r'visa',
            r'mastercard',
            r'master\s+card',
            r'amex',
            r'american\s+express',
            r'discover',
            r'ebt',
            r'redcard',
            r'payment',
            r'tender',
            r'ref\s*[#:]',
            r'tc\s*[#:]',
            r'receipt\s*[#:]',
            r'thank\s+you',
            r'savings',
            r'discount',
            r'member',
            r'store\s*[#:]',
            r'warehouse\s*[#:]',
            r'cashier',
            r'register',
            r'lane',
            r'^$',  # Empty lines
            r'^\s*[*\-=]+\s*$',  # Lines with only symbols
        ]
        
        for pattern in exclusion_patterns:
            if re.search(pattern, line_lower):
                return True
        
        return False
    
    def extract_data(self, text: str) -> Dict[str, Any]:
        """Extract structured data from the receipt text using this template."""
        lines = text.strip().split('\n')
        
        result = {
            'vendor': self.vendor_name,
            'vendor_id': self.vendor_id,
            'store_name': self.vendor_name,
            'items': [],
            'template_match': True
        }
        
        # Extract header information
        for key, pattern in self.header_patterns.items():
            for line in lines:
                match = pattern.search(line)
                if match:
                    result[key] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract address
        if self.address_pattern:
            for i, line in enumerate(lines):
                match = self.address_pattern.search(line)
                if match:
                    result['store_address'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract phone
        if self.phone_pattern:
            for line in lines:
                match = self.phone_pattern.search(line)
                if match:
                    result['phone_number'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract date and time
        if self.date_pattern:
            for line in lines:
                match = self.date_pattern.search(line)
                if match:
                    result['date'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        if self.time_pattern:
            for line in lines:
                match = self.time_pattern.search(line)
                if match:
                    result['time'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract receipt number
        if self.receipt_number_pattern:
            for line in lines:
                match = self.receipt_number_pattern.search(line)
                if match:
                    result['receipt_number'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract items with improved filtering
        if self.item_pattern:
            for line in lines:
                # Skip lines that should be excluded
                if self.is_excluded_line(line):
                    continue
                
                match = self.item_pattern.search(line)
                if match and len(match.groups()) >= 2:
                    item_name = match.group(1).strip()
                    item_price = match.group(2).strip()
                    
                    # Additional filtering for item names
                    if len(item_name) > 2 and not self.is_excluded_line(item_name):
                        # Optional quantity if available
                        quantity = match.group(3).strip() if len(match.groups()) >= 3 and match.group(3) else "1"
                        
                        result['items'].append({
                            'name': item_name,
                            'price': item_price,
                            'quantity': quantity
                        })
        
        # Extract total amounts
        if self.subtotal_pattern:
            for line in lines:
                match = self.subtotal_pattern.search(line)
                if match:
                    result['subtotal'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        if self.tax_pattern:
            for line in lines:
                match = self.tax_pattern.search(line)
                if match:
                    result['tax'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        if self.total_pattern:
            for line in lines:
                match = self.total_pattern.search(line)
                if match:
                    result['total'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        # Extract payment method
        if self.payment_method_pattern:
            for line in lines:
                match = self.payment_method_pattern.search(line)
                if match:
                    result['payment_method'] = match.group(1).strip() if match.groups() else line.strip()
                    break
        
        return result

    @classmethod
    def from_dict(cls, data: Dict) -> 'ReceiptTemplate':
        """Create a template from a dictionary."""
        return cls(
            vendor_id=data.get('vendor_id', ''),
            vendor_name=data.get('vendor_name', ''),
            recognition_patterns=data.get('recognition_patterns', []),
            header_patterns=data.get('header_patterns', {}),
            item_pattern=data.get('item_pattern', None),
            total_pattern=data.get('total_pattern', None),
            subtotal_pattern=data.get('subtotal_pattern', None),
            tax_pattern=data.get('tax_pattern', None),
            date_pattern=data.get('date_pattern', None),
            time_pattern=data.get('time_pattern', None),
            receipt_number_pattern=data.get('receipt_number_pattern', None),
            payment_method_pattern=data.get('payment_method_pattern', None),
            address_pattern=data.get('address_pattern', None),
            phone_pattern=data.get('phone_pattern', None),
        )
    
    def to_dict(self) -> Dict:
        """Convert template to a dictionary."""
        return {
            'vendor_id': self.vendor_id,
            'vendor_name': self.vendor_name,
            'recognition_patterns': [p.pattern for p in self.recognition_patterns],
            'header_patterns': {k: v.pattern for k, v in self.header_patterns.items()},
            'item_pattern': self.item_pattern.pattern if self.item_pattern else None,
            'total_pattern': self.total_pattern.pattern if self.total_pattern else None,
            'subtotal_pattern': self.subtotal_pattern.pattern if self.subtotal_pattern else None,
            'tax_pattern': self.tax_pattern.pattern if self.tax_pattern else None,
            'date_pattern': self.date_pattern.pattern if self.date_pattern else None,
            'time_pattern': self.time_pattern.pattern if self.time_pattern else None,
            'receipt_number_pattern': self.receipt_number_pattern.pattern if self.receipt_number_pattern else None,
            'payment_method_pattern': self.payment_method_pattern.pattern if self.payment_method_pattern else None,
            'address_pattern': self.address_pattern.pattern if self.address_pattern else None,
            'phone_pattern': self.phone_pattern.pattern if self.phone_pattern else None,
        }
    
    def save(self, directory: str) -> str:
        """Save template to a JSON file."""
        filename = os.path.join(directory, f"{self.vendor_id}.json")
        with open(filename, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        return filename


class TemplateManager:
    """Manager for receipt templates."""
    
    def __init__(self, templates_dir: str = "templates/vendors"):
        """
        Initialize the template manager.
        
        Args:
            templates_dir: Directory containing template JSON files
        """
        self.templates_dir = templates_dir
        self.templates = []
        self.load_templates()
    
    def load_templates(self) -> None:
        """Load all templates from the templates directory."""
        if not os.path.exists(self.templates_dir):
            os.makedirs(self.templates_dir, exist_ok=True)
            return
        
        for filename in os.listdir(self.templates_dir):
            if filename.endswith('.json'):
                filepath = os.path.join(self.templates_dir, filename)
                try:
                    with open(filepath, 'r') as f:
                        template_data = json.load(f)
                        template = ReceiptTemplate.from_dict(template_data)
                        self.templates.append(template)
                except Exception as e:
                    print(f"Error loading template {filename}: {e}")
    
    def add_template(self, template: ReceiptTemplate) -> None:
        """Add a new template to the manager."""
        # Check if template with this ID already exists
        for i, t in enumerate(self.templates):
            if t.vendor_id == template.vendor_id:
                # Replace existing template
                self.templates[i] = template
                template.save(self.templates_dir)
                return
        
        # Add new template
        self.templates.append(template)
        template.save(self.templates_dir)
    
    def find_matching_template(self, text: str) -> Optional[ReceiptTemplate]:
        """Find a template that matches the given text."""
        for template in self.templates:
            if template.matches(text):
                return template
        return None
    
    def parse_receipt(self, text: str) -> Dict[str, Any]:
        """Parse a receipt using the appropriate template."""
        template = self.find_matching_template(text)
        if template:
            return template.extract_data(text)
        
        # Return empty result if no template matches
        return {
            'template_match': False,
            'items': []
        }
    
    def get_template_by_id(self, vendor_id: str) -> Optional[ReceiptTemplate]:
        """Get a template by vendor ID."""
        for template in self.templates:
            if template.vendor_id == vendor_id:
                return template
        return None
    
    def get_all_templates(self) -> List[Dict[str, str]]:
        """Get a list of all templates with basic info."""
        return [
            {'id': t.vendor_id, 'name': t.vendor_name}
            for t in self.templates
        ]
    
    def delete_template(self, vendor_id: str) -> bool:
        """Delete a template by vendor ID."""
        for i, template in enumerate(self.templates):
            if template.vendor_id == vendor_id:
                filepath = os.path.join(self.templates_dir, f"{vendor_id}.json")
                if os.path.exists(filepath):
                    os.remove(filepath)
                del self.templates[i]
                return True
        return False 