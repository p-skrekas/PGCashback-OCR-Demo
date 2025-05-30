import streamlit as st
from PIL import Image, ImageDraw, ImageFont
import io
import json
import base64
import os
import re
import numpy as np
import cv2
from google.cloud import vision
from google.oauth2 import service_account
import vertexai
from vertexai.generative_models import GenerativeModel, Part

def setup_google_vision():
    """Initialize Google Vision client with credentials from secrets or local file"""
    try:
        # Try to load from Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            # Load credentials from Streamlit secrets
            credentials_info = dict(st.secrets['gcp_service_account'])
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        else:
            # Fallback to local file (for development)
            credentials_path = "pg-cashback-5290642eb30b.json"
            
            if not os.path.exists(credentials_path):
                # Check if we're in a cloud environment (no local file expected)
                if 'streamlit' in str(os.environ.get('HOME', '')).lower() or 'app' in str(os.environ.get('HOME', '')).lower():
                    st.error("Streamlit secrets not configured properly")
                    st.info("Please configure Google Cloud credentials in the app secrets.")
                else:
                    st.error("Local development: Credentials file not found")
                    st.info("For local development, ensure your Google Cloud credentials file is present.")
                return None
            
            # Create credentials from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
        
        # Initialize the Vision client with credentials
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
        
    except Exception as e:
        st.error(f"Error initializing Google Vision client: {str(e)}")
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            st.info("Please check your Streamlit secrets configuration.")
        else:
            st.info("Please ensure your Google Cloud credentials are properly configured.")
        return None

def setup_vertex_ai():
    """Initialize Vertex AI with credentials from secrets or local file"""
    try:
        # Try to load from Streamlit secrets first (for cloud deployment)
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            # Load credentials from Streamlit secrets
            credentials_info = dict(st.secrets['gcp_service_account'])
            credentials = service_account.Credentials.from_service_account_info(
                credentials_info,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            project_id = credentials_info['project_id']
        else:
            # Fallback to local file (for development)
            credentials_path = "pg-cashback-5290642eb30b.json"
            
            if not os.path.exists(credentials_path):
                # Check if we're in a cloud environment (no local file expected)
                if 'streamlit' in str(os.environ.get('HOME', '')).lower() or 'app' in str(os.environ.get('HOME', '')).lower():
                    st.error("Streamlit secrets not configured properly")
                    st.info("Please configure Google Cloud credentials in the app secrets.")
                else:
                    st.error("Local development: Credentials file not found")
                    st.info("For local development, ensure your Google Cloud credentials file is present.")
                return None
            
            # Create credentials from the service account file
            credentials = service_account.Credentials.from_service_account_file(
                credentials_path,
                scopes=['https://www.googleapis.com/auth/cloud-platform']
            )
            
            # Load project ID from file
            with open(credentials_path, 'r') as f:
                cred_data = json.load(f)
            project_id = cred_data['project_id']
        
        # Initialize Vertex AI
        vertexai.init(
            project=project_id,
            location="us-central1",  # You can change this to your preferred region
            credentials=credentials
        )
        
        # Initialize the Gemini model
        model = GenerativeModel('gemini-2.0-flash-exp')
        return model
        
    except Exception as e:
        st.error(f"Error initializing Vertex AI: {str(e)}")
        if hasattr(st, 'secrets') and 'gcp_service_account' in st.secrets:
            st.info("Please check your Streamlit secrets configuration and ensure Vertex AI API is enabled.")
        else:
            st.info("Please ensure Vertex AI API is enabled and credentials are configured.")
        return None

def extract_text_with_vision(image_bytes, client):
    """Extract text from image using Google Vision OCR"""
    try:
        # Create Vision Image object
        image = vision.Image(content=image_bytes)
        
        # Perform text detection
        response = client.text_detection(image=image)
        texts = response.text_annotations
        
        if response.error.message:
            raise Exception(f'{response.error.message}')
        
        if texts:
            # Return both the full text and the individual text annotations with bounding boxes
            full_text = texts[0].description
            # Skip the first annotation which contains all text
            text_annotations = texts[1:]
            return full_text, text_annotations
        else:
            return "No text detected in the image.", []
            
    except Exception as e:
        st.error(f"Error during OCR processing: {str(e)}")
        return None, []

def draw_bounding_boxes(image_bytes, text_annotations):
    """Draw bounding boxes around detected text in the image"""
    try:
        # Convert image bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Convert to RGB (OpenCV uses BGR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get original dimensions
        original_height, original_width = img.shape[:2]
        
        # Calculate scaling factor to keep reasonable size (max 800px width)
        max_width = 800
        if original_width > max_width:
            scale_factor = max_width / original_width
            new_width = max_width
            new_height = int(original_height * scale_factor)
            img = cv2.resize(img, (new_width, new_height))
        else:
            scale_factor = 1.0
        
        # Draw bounding boxes (scaled if necessary)
        for text in text_annotations:
            points = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            
            # Scale the coordinates if image was resized
            if scale_factor != 1.0:
                points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]
            
            # Create a rectangle (OpenCV requires integers for rectangle points)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Draw the rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add text description with smaller font
            # Display a truncated version if the text is too long
            display_text = text.description[:12] + "..." if len(text.description) > 12 else text.description
            font_scale = 0.4 if scale_factor < 1.0 else 0.5
            cv2.putText(img, display_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 1)
        
        # Convert back to PIL Image
        return Image.fromarray(img)
    
    except Exception as e:
        st.error(f"Error drawing bounding boxes: {str(e)}")
        return None

def extract_structured_data(text):
    """Extract structured data from receipt text"""
    if not text:
        return None
    
    lines = [line.strip() for line in text.strip().split('\n') if line.strip()]
    
    structured_data = {
        'store_name': None,
        'store_address': None,
        'phone_number': None,
        'date': None,
        'time': None,
        'receipt_number': None,
        'items': [],
        'subtotal': None,
        'tax': None,
        'total': None,
        'payment_method': None
    }
    
    # Patterns for different data types
    patterns = {
        'phone': r'(\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4})',
        'date': r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
        'time': r'(\d{1,2}:\d{2}(?::\d{2})?(?:\s?[AP]M)?)',
        'amount': r'(\$?\d+[.,]\d{2})',
        'receipt_num': r'(?:receipt|rcpt|ref|#)\s*:?\s*([A-Z0-9]+)',
        'item_price': r'^(.+?)\s+(\$?\d+[.,]\d{2})$'
    }
    
    # Extract store info (usually first few lines)
    if lines:
        structured_data['store_name'] = lines[0]
        if len(lines) > 1:
            # Look for address-like patterns
            for i in range(1, min(4, len(lines))):
                if any(word in lines[i].lower() for word in ['street', 'st', 'ave', 'road', 'rd', 'blvd']):
                    structured_data['store_address'] = lines[i]
                    break
    
    # Process each line
    for i, line in enumerate(lines):
        line_lower = line.lower()
        
        # Phone number
        if not structured_data['phone_number']:
            phone_match = re.search(patterns['phone'], line)
            if phone_match:
                structured_data['phone_number'] = phone_match.group(1)
        
        # Date
        if not structured_data['date']:
            date_match = re.search(patterns['date'], line)
            if date_match:
                structured_data['date'] = date_match.group(1)
        
        # Time
        if not structured_data['time']:
            time_match = re.search(patterns['time'], line)
            if time_match:
                structured_data['time'] = time_match.group(1)
        
        # Receipt number
        if not structured_data['receipt_number']:
            receipt_match = re.search(patterns['receipt_num'], line, re.IGNORECASE)
            if receipt_match:
                structured_data['receipt_number'] = receipt_match.group(1)
        
        # Totals
        if any(keyword in line_lower for keyword in ['subtotal', 'sub total']):
            amount_match = re.search(patterns['amount'], line)
            if amount_match:
                structured_data['subtotal'] = amount_match.group(1)
        
        elif any(keyword in line_lower for keyword in ['tax', 'hst', 'gst', 'vat']):
            amount_match = re.search(patterns['amount'], line)
            if amount_match:
                structured_data['tax'] = amount_match.group(1)
        
        elif any(keyword in line_lower for keyword in ['total', 'amount due', 'balance']):
            amount_match = re.search(patterns['amount'], line)
            if amount_match:
                structured_data['total'] = amount_match.group(1)
        
        # Payment method
        elif any(keyword in line_lower for keyword in ['visa', 'mastercard', 'amex', 'cash', 'debit', 'credit']):
            structured_data['payment_method'] = line
        
        # Items (lines with item name and price)
        else:
            item_match = re.search(patterns['item_price'], line)
            if item_match and not any(keyword in line_lower for keyword in ['tax', 'total', 'subtotal', 'change']):
                item_name = item_match.group(1).strip()
                item_price = item_match.group(2)
                if len(item_name) > 2:  # Avoid single characters or very short strings
                    structured_data['items'].append({
                        'name': item_name,
                        'price': item_price
                    })
    
    return structured_data

def analyze_receipt_text(text):
    """Analyze extracted text to identify receipt components"""
    if not text:
        return None
    
    lines = text.strip().split('\n')
    
    analysis = {
        'total_lines': len(lines),
        'raw_text': text,
        'lines': lines,
        'potential_amounts': [],
        'potential_dates': [],
        'potential_store_info': []
    }
    
    # Look for monetary amounts (simple pattern)
    amount_pattern = r'\$?\d+[.,]\d{2}'
    date_pattern = r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}'
    
    for line in lines:
        # Find potential amounts
        amounts = re.findall(amount_pattern, line)
        if amounts:
            analysis['potential_amounts'].extend(amounts)
        
        # Find potential dates
        dates = re.findall(date_pattern, line)
        if dates:
            analysis['potential_dates'].extend(dates)
    
    # Store info is typically in the first few lines
    analysis['potential_store_info'] = lines[:3]
    
    return analysis

def analyze_text_with_llm(extracted_text, model):
    """Analyze OCR-extracted text using Vertex AI Gemini model"""
    try:
        # Create the prompt for text analysis
        prompt = f"""
        Analyze this receipt text that was extracted using OCR and structure the following information in JSON format. 
        Be as accurate as possible and only include information that is clearly present in the text.
        
        Receipt Text:
        {extracted_text}
        
        Please extract and structure:
        {{
            "store_name": "Name of the store/business",
            "store_address": "Complete address if visible",
            "phone_number": "Phone number if visible",
            "date": "Date of transaction (MM/DD/YYYY or DD/MM/YYYY format)",
            "time": "Time of transaction",
            "receipt_number": "Receipt or transaction number",
            "items": [
                {{
                    "name": "Item name",
                    "price": "Item price",
                    "quantity": "Quantity if specified"
                }}
            ],
            "subtotal": "Subtotal amount",
            "tax": "Tax amount",
            "total": "Total amount",
            "payment_method": "Payment method used (cash, card, etc.)",
            "cashier": "Cashier name if visible",
            "additional_info": "Any other relevant information like discounts, promotions, etc."
        }}
        
        If any field is not clearly present in the text, set it to null.
        Make sure to return valid JSON format.
        """
        
        # Generate content with the text prompt
        response = model.generate_content(prompt)
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                return parsed_data, response_text
            else:
                # If no JSON found, return the raw response
                return None, response_text
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return None, response_text
            
    except Exception as e:
        st.error(f"Error analyzing text with LLM: {str(e)}")
        return None, None

def analyze_receipt_with_llm(image_bytes, model):
    """Analyze receipt image directly using Vertex AI Gemini model"""
    try:
        # Convert image bytes to base64 for Vertex AI
        b64_image = base64.b64encode(image_bytes).decode('utf-8')
        mime_type = "image/jpeg"  # Assuming JPEG, adjust if needed
        
        # Create image part for Vertex AI
        image_part = Part.from_data(
            data=base64.b64decode(b64_image), 
            mime_type=mime_type
        )
        
        # Create the prompt for receipt analysis
        prompt = """
        Analyze this receipt image directly and extract the following information in JSON format. 
        Be as accurate as possible and only include information that is clearly visible in the receipt.
        
        Please extract:
        {
            "store_name": "Name of the store/business",
            "store_address": "Complete address if visible",
            "phone_number": "Phone number if visible",
            "date": "Date of transaction (MM/DD/YYYY or DD/MM/YYYY format)",
            "time": "Time of transaction",
            "receipt_number": "Receipt or transaction number",
            "items": [
                {
                    "name": "Item name",
                    "price": "Item price",
                    "quantity": "Quantity if specified"
                }
            ],
            "subtotal": "Subtotal amount",
            "tax": "Tax amount",
            "total": "Total amount",
            "payment_method": "Payment method used (cash, card, etc.)",
            "cashier": "Cashier name if visible",
            "additional_info": "Any other relevant information like discounts, promotions, etc."
        }
        
        If any field is not clearly visible or readable in the receipt, set it to null.
        Make sure to return valid JSON format.
        """
        
        # Generate content with the image and prompt
        response = model.generate_content([prompt, image_part])
        
        # Parse the response
        response_text = response.text.strip()
        
        # Try to extract JSON from the response
        try:
            # Look for JSON in the response
            json_start = response_text.find('{')
            json_end = response_text.rfind('}') + 1
            
            if json_start != -1 and json_end != -1:
                json_str = response_text[json_start:json_end]
                parsed_data = json.loads(json_str)
                return parsed_data, response_text
            else:
                # If no JSON found, return the raw response
                return None, response_text
                
        except json.JSONDecodeError:
            # If JSON parsing fails, return the raw response
            return None, response_text
            
    except Exception as e:
        st.error(f"Error analyzing receipt with Direct LLM: {str(e)}")
        return None, None

def main():
    # Page configuration
    st.set_page_config(
        page_title="Receipt Scanner: OCR+LLM vs Direct LLM",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS for professional styling
    st.markdown("""
    <style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styling */
    .main > div {
        padding-top: 2rem;
    }
    
    /* Custom styling for the main app */
    .stApp {
        font-family: 'Inter', sans-serif;
        color: #1f2937;
    }
    
    /* Ensure black text on white backgrounds */
    .main .block-container {
        color: #1f2937;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        text-align: center;
        color: white;
    }
    
    .main-header h1 {
        color: white !important;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .main-header p {
        color: rgba(255, 255, 255, 0.9);
        font-size: 1.2rem;
        margin: 0;
    }
    
    /* Status cards */
    .status-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
        color: #1f2937;
    }
    
    .status-card h4 {
        color: #1f2937 !important;
        margin-bottom: 0.5rem;
    }
    
    .status-card p {
        color: #374151 !important;
        margin: 0.25rem 0;
    }
    
    .status-card.success {
        border-left-color: #10b981;
    }
    
    .status-card.error {
        border-left-color: #ef4444;
    }
    
    /* Analysis buttons */
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        font-size: 1rem;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(102, 126, 234, 0.4);
    }
    
    /* Metrics styling */
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        color: #1f2937;
    }
    
    .metric-container h4 {
        color: #1f2937 !important;
        margin-bottom: 0.5rem;
    }
    
    .metric-container p {
        color: #374151 !important;
        margin: 0;
    }
    
    /* Section headers */
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937 !important;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #e5e7eb;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
        background-color: #f8fafc;
        border-radius: 10px;
        padding: 0.5rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.5rem;
        background-color: transparent;
        border-radius: 8px;
        font-weight: 500;
        color: #374151 !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white !important;
    }
    
    /* Login form styling */
    .login-container {
        background: white;
        padding: 3rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        max-width: 400px;
        margin: 2rem auto;
        color: #1f2937;
    }
    
    .login-container h3 {
        color: #1f2937 !important;
    }
    
    .login-container p {
        color: #374151 !important;
    }
    
    .login-container * {
        color: #1f2937 !important;
    }
    
    /* Login form input fields */
    .login-container .stTextInput label {
        color: #1f2937 !important;
        font-weight: 500;
    }
    
    .login-container .stTextInput input {
        color: #1f2937 !important;
        background-color: white !important;
        border: 1px solid #d1d5db;
    }
    
    .login-container .stTextInput input::placeholder {
        color: #9ca3af !important;
    }
    
    /* Login form markdown text */
    .login-container .stMarkdown {
        color: #1f2937 !important;
    }
    
    .login-container .stMarkdown * {
        color: #1f2937 !important;
    }
    
    /* Image display styling */
    .image-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        text-align: center;
        color: #1f2937;
    }
    
    /* Radio button styling */
    .stRadio > div {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e5e7eb;
        color: #1f2937;
    }
    
    .stRadio label {
        color: #1f2937 !important;
    }
    
    .stRadio div[role="radiogroup"] label {
        color: #1f2937 !important;
        font-weight: 500;
    }
    
    .stRadio div[role="radiogroup"] div {
        color: #1f2937 !important;
    }
    
    /* Image input section specific styling */
    .stRadio span {
        color: #1f2937 !important;
    }
    
    /* Ensure all radio button text is dark */
    .stRadio * {
        color: #1f2937 !important;
    }
    
    /* File uploader section */
    .stFileUploader {
        color: #1f2937;
    }
    
    .stFileUploader > div {
        color: #1f2937 !important;
    }
    
    .stFileUploader span {
        color: #1f2937 !important;
    }
    
    /* Camera input section */
    .stCameraInput {
        color: #1f2937;
    }
    
    .stCameraInput > div {
        color: #1f2937 !important;
    }
    
    .stCameraInput span {
        color: #1f2937 !important;
    }
    
    /* Selectbox for sample images */
    .stSelectbox {
        color: #1f2937;
    }
    
    .stSelectbox > div {
        color: #1f2937 !important;
    }
    
    .stSelectbox span {
        color: #1f2937 !important;
    }
    
    /* Markdown text in image input section */
    .stMarkdown h4 {
        color: #1f2937 !important;
        font-weight: 600;
    }
    
    /* Any text elements in the main content area */
    .main .block-container p {
        color: #374151 !important;
    }
    
    .main .block-container span {
        color: #1f2937 !important;
    }
    
    /* Info boxes */
    .info-box {
        background: linear-gradient(135deg, #667eea, #764ba2);
        color: white;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .info-box h4 {
        color: white !important;
        margin-bottom: 0.5rem;
    }
    
    .info-box p {
        color: rgba(255, 255, 255, 0.9) !important;
        margin: 0;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Text areas and content */
    .stTextArea textarea {
        color: #1f2937 !important;
        background-color: white !important;
    }
    
    /* Raw data tab specific styling */
    .stTextArea label {
        color: #1f2937 !important;
        font-weight: 500;
    }
    
    /* Text area content */
    textarea {
        color: #1f2937 !important;
        background-color: white !important;
        border: 1px solid #d1d5db;
    }
    
    /* Markdown content */
    .stMarkdown {
        color: #1f2937;
    }
    
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h5, .stMarkdown h6 {
        color: #1f2937 !important;
    }
    
    .stMarkdown p {
        color: #374151 !important;
    }
    
    /* Raw data section headers */
    .stMarkdown h4 {
        color: #1f2937 !important;
        font-weight: 600;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    .css-1d391kg .stMarkdown {
        color: white !important;
    }
    
    .css-1d391kg .stMarkdown h1, 
    .css-1d391kg .stMarkdown h2, 
    .css-1d391kg .stMarkdown h3, 
    .css-1d391kg .stMarkdown h4, 
    .css-1d391kg .stMarkdown h5, 
    .css-1d391kg .stMarkdown h6 {
        color: white !important;
    }
    
    .css-1d391kg .stMarkdown p {
        color: rgba(255, 255, 255, 0.9) !important;
    }
    
    /* Download buttons */
    .download-section {
        background: #f8fafc;
        padding: 2rem;
        border-radius: 10px;
        margin-top: 2rem;
        color: #1f2937;
    }
    
    .download-section p {
        color: #374151 !important;
    }
    
    /* Expander content */
    .streamlit-expanderContent {
        background-color: white;
        color: #1f2937;
    }
    
    .streamlit-expanderContent p {
        color: #374151 !important;
    }
    
    /* Success/Error messages */
    .stSuccess {
        color: #10b981 !important;
    }
    
    .stError {
        color: #ef4444 !important;
    }
    
    .stInfo {
        color: #3b82f6 !important;
    }
    
    .stWarning {
        color: #f59e0b !important;
    }
    
    /* File uploader */
    .stFileUploader label {
        color: #1f2937 !important;
    }
    
    /* Camera input */
    .stCameraInput label {
        color: #1f2937 !important;
    }
    
    /* Selectbox */
    .stSelectbox label {
        color: #1f2937 !important;
    }
    
    /* Form elements */
    .stTextInput label {
        color: #1f2937 !important;
    }
    
    .stTextInput input {
        color: #1f2937 !important;
        background-color: white !important;
    }
    
    </style>
    """, unsafe_allow_html=True)
    
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Login screen with professional styling
        st.markdown("""
        <div class="main-header">
            <h1>🔐 Welcome to Receipt Scanner Pro</h1>
            <p>Advanced AI-Powered Receipt Analysis Platform</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Center the login form
        col1, col2, col3 = st.columns([1, 1, 1])
        
        with col2:
            st.markdown('<div class="login-container">', unsafe_allow_html=True)
            
            with st.form("login_form"):
                st.markdown("### 🔑 Authentication Required")
                st.markdown("Please enter your credentials to access the platform")
                
                username = st.text_input("👤 Username", placeholder="Enter username")
                password = st.text_input("🔒 Password", type="password", placeholder="Enter password")
                
                st.markdown("<br>", unsafe_allow_html=True)
                login_button = st.form_submit_button("🚀 Access Platform", use_container_width=True)
                
                if login_button:
                    if username == "reborrn" and password == "password":
                        st.session_state.authenticated = True
                        st.success("✅ Authentication successful! Welcome to Receipt Scanner Pro")
                        st.rerun()
                    else:
                        st.error("❌ Invalid credentials. Please try again.")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Footer
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.markdown(
            """
            <div style='text-align: center; color: #6b7280; font-size: 14px; padding: 2rem;'>
                <p>🧾 <strong>Receipt Scanner Pro</strong> - Compare OCR+LLM vs Direct LLM approaches</p>
                <p>Powered by Google Cloud Vision API & Vertex AI Gemini</p>
            </div>
            """, 
            unsafe_allow_html=True
        )
        return
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown("### 👤 User Dashboard")
        st.markdown("**Logged in as:** reborrn")
        st.markdown("**Role:** Administrator")
        st.markdown("---")
        
        st.markdown("### 📊 Session Info")
        if hasattr(st.session_state, 'ocr_llm_structured_data') or hasattr(st.session_state, 'direct_llm_structured_data'):
            st.success("✅ Analysis data available")
        else:
            st.info("🔄 No analysis performed yet")
        
        st.markdown("---")
        if st.button("🚪 Logout", use_container_width=True):
            st.session_state.authenticated = False
            # Clear all other session state variables
            for key in list(st.session_state.keys()):
                if key != 'authenticated':
                    del st.session_state[key]
            st.rerun()
    
    # Professional main header
    st.markdown("""
    <div class="main-header">
        <h1>🧾 Receipt Scanner Pro</h1>
        <p>Advanced OCR+LLM vs Direct LLM Comparison Platform</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize clients with professional status display
    st.markdown('<div class="section-header">🔧 System Status</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        vision_client = setup_google_vision()
        if vision_client:
            st.markdown("""
            <div class="status-card success">
                <h4>🔍 OCR Pipeline</h4>
                <p><strong>Status:</strong> <span style="color: #10b981;">✅ Online</span></p>
                <p><strong>Service:</strong> Google Vision API</p>
                <p><strong>Capability:</strong> Text extraction with bounding boxes</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card error">
                <h4>🔍 OCR Pipeline</h4>
                <p><strong>Status:</strong> <span style="color: #ef4444;">❌ Offline</span></p>
                <p><strong>Service:</strong> Google Vision API</p>
                <p><strong>Issue:</strong> Configuration required</p>
            </div>
            """, unsafe_allow_html=True)
    
    with col2:
        vertex_model = setup_vertex_ai()
        if vertex_model:
            st.markdown("""
            <div class="status-card success">
                <h4>🤖 AI Analysis Engine</h4>
                <p><strong>Status:</strong> <span style="color: #10b981;">✅ Online</span></p>
                <p><strong>Service:</strong> Vertex AI Gemini 2.0 Flash</p>
                <p><strong>Capability:</strong> Direct image analysis & text processing</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="status-card error">
                <h4>🤖 AI Analysis Engine</h4>
                <p><strong>Status:</strong> <span style="color: #ef4444;">❌ Offline</span></p>
                <p><strong>Service:</strong> Vertex AI Gemini 2.0 Flash</p>
                <p><strong>Issue:</strong> Configuration required</p>
            </div>
            """, unsafe_allow_html=True)
    
    if not vision_client and not vertex_model:
        st.warning("⚠️ No analysis methods available. Please check your setup.")
        return
    
    # File upload section with professional styling
    st.markdown('<div class="section-header">📷 Image Input</div>', unsafe_allow_html=True)
    
    # Enhanced image source selection
    st.markdown("Choose your preferred image input method:")
    image_source = st.radio(
        "",
        ["📁 Upload from device", "📸 Take photo", "🗂️ Sample images"],
        horizontal=True,
        label_visibility="collapsed"
    )
    
    uploaded_file = None
    image_bytes = None
    image = None
    
    if image_source == "📁 Upload from device":
        st.markdown("#### Upload Receipt Image")
        uploaded_file = st.file_uploader(
            "Choose a receipt image...",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG | Max size: 200MB"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_bytes = uploaded_file.getvalue()
    
    elif image_source == "📸 Take photo":
        st.markdown("#### Camera Capture")
        st.markdown("📱 Position your device camera over the receipt and capture")
        camera_photo = st.camera_input("Take a picture")
        
        if camera_photo is not None:
            image = Image.open(camera_photo)
            image_bytes = camera_photo.getvalue()
            
            # Create a mock uploaded_file object for compatibility
            class MockUploadedFile:
                def __init__(self, filename, file_bytes):
                    self.name = filename
                    self.size = len(file_bytes)
            
            uploaded_file = MockUploadedFile("camera_photo.jpg", image_bytes)
    
    else:  # Sample images
        # Check if receipts/images folder exists
        sample_folder = os.path.join("receipts", "images")
        if os.path.exists(sample_folder):
            # Get list of image files
            image_files = []
            for ext in ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']:
                image_files.extend([f for f in os.listdir(sample_folder) if f.endswith(ext.replace('*', ''))])
            
            if image_files:
                selected_file = st.selectbox(
                    "Select a sample receipt:",
                    [""] + sorted(image_files),
                    help="Choose from available sample receipt images"
                )
                
                if selected_file:
                    file_path = os.path.join(sample_folder, selected_file)
                    try:
                        image = Image.open(file_path)
                        with open(file_path, 'rb') as f:
                            image_bytes = f.read()
                        
                        # Create a mock uploaded_file object for compatibility
                        class MockUploadedFile:
                            def __init__(self, filename, file_bytes):
                                self.name = filename
                                self.size = len(file_bytes)
                        
                        uploaded_file = MockUploadedFile(selected_file, image_bytes)
                        
                    except Exception as e:
                        st.error(f"Error loading sample image: {str(e)}")
                        image = None
                        image_bytes = None
            else:
                st.info("No sample images found in receipts/images folder")
        else:
            st.info("Sample images folder (receipts/images) not found")
    
    if image is not None and image_bytes is not None:
        try:
            # Professional image display
            st.markdown('<div class="section-header">📊 Image Analysis</div>', unsafe_allow_html=True)
            
            # Create tabs for viewing options
            image_tab1, image_tab2 = st.tabs(["🖼️ Original Image", "🔍 OCR Visualization"])
            
            with image_tab1:
                st.markdown('<div class="image-container">', unsafe_allow_html=True)
                st.image(image, caption="Receipt Image", width=600)
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Professional image metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📄 Filename</h4>
                        <p>{uploaded_file.name}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col2:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>💾 File Size</h4>
                        <p>{len(image_bytes) / 1024:.1f} KB</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col3:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>📐 Dimensions</h4>
                        <p>{image.size[0]} × {image.size[1]}</p>
                    </div>
                    """, unsafe_allow_html=True)
                with col4:
                    st.markdown(f"""
                    <div class="metric-container">
                        <h4>🎨 Format</h4>
                        <p>{image.format}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Analysis section with professional styling
            st.markdown('<div class="section-header">🚀 AI Analysis</div>', unsafe_allow_html=True)
            st.markdown("Choose your analysis method to process the receipt:")
            
            # Professional analysis buttons
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                if vision_client and vertex_model:
                    if st.button("🔍➡️🤖 OCR + LLM Pipeline", type="primary", use_container_width=True):
                        with st.spinner("🔄 Step 1: Extracting text with Google Vision OCR..."):
                            # Extract text using Google Vision
                            extracted_text, text_annotations = extract_text_with_vision(image_bytes, vision_client)
                            
                            if extracted_text:
                                # Create the visualization image with bounding boxes
                                if text_annotations:
                                    visualization_image = draw_bounding_boxes(image_bytes, text_annotations)
                                    if visualization_image:
                                        st.session_state.ocr_visualization_image = visualization_image
                                
                                with st.spinner("🔄 Step 2: Analyzing extracted text with Gemini..."):
                                    # Analyze the extracted text with LLM
                                    ocr_llm_structured_data, ocr_llm_raw_response = analyze_text_with_llm(extracted_text, vertex_model)
                                    
                                    if ocr_llm_structured_data or ocr_llm_raw_response:
                                        # Store OCR+LLM results in session state
                                        st.session_state.ocr_extracted_text = extracted_text
                                        st.session_state.ocr_llm_structured_data = ocr_llm_structured_data
                                        st.session_state.ocr_llm_raw_response = ocr_llm_raw_response
                                        
                                        st.success("✅ OCR+LLM pipeline analysis completed!")
                                    else:
                                        st.error("❌ Failed to analyze extracted text with LLM.")
                            else:
                                st.error("❌ Failed to extract text with OCR.")
                else:
                    st.button("🔍➡️🤖 OCR + LLM Pipeline", disabled=True, use_container_width=True)
                    st.caption("⚠️ Requires both Vision API and Vertex AI")
            
            with analysis_col2:
                if vertex_model:
                    if st.button("🤖 Direct LLM Analysis", type="secondary", use_container_width=True):
                        with st.spinner("🔄 Processing image directly with Gemini..."):
                            # Analyze using Direct LLM
                            direct_llm_structured_data, direct_llm_raw_response = analyze_receipt_with_llm(image_bytes, vertex_model)
                            
                            if direct_llm_structured_data or direct_llm_raw_response:
                                # Store Direct LLM results in session state
                                st.session_state.direct_llm_structured_data = direct_llm_structured_data
                                st.session_state.direct_llm_raw_response = direct_llm_raw_response
                                
                                st.success("✅ Direct LLM analysis completed!")
                            else:
                                st.error("❌ Failed to analyze with Direct LLM.")
                else:
                    st.button("🤖 Direct LLM Analysis", disabled=True, use_container_width=True)
                    st.caption("⚠️ Vertex AI not available")
            
            # Display OCR visualization in the second tab if available
            with image_tab2:
                st.markdown("#### 🔍 OCR Text Detection")
                if hasattr(st.session_state, 'ocr_visualization_image') and st.session_state.ocr_visualization_image is not None:
                    st.markdown('<div class="image-container">', unsafe_allow_html=True)
                    st.image(st.session_state.ocr_visualization_image, caption="Receipt with OCR Text Detection", width=600)
                    st.markdown('</div>', unsafe_allow_html=True)
                    st.info("🔎 Green boxes show detected text regions with labels")
                else:
                    st.info("👆 Click 'OCR + LLM Pipeline' to see text detection visualization")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the file format.")
    
    else:
        # Professional instructions when no image is selected
        st.markdown('<div class="section-header">🚀 Get Started</div>', unsafe_allow_html=True)
        
        if image_source == "📁 Upload from device":
            st.markdown("""
            <div class="info-box">
                <h4>📁 Upload Receipt Image</h4>
                <p>Select a receipt image from your device to begin analysis</p>
            </div>
            """, unsafe_allow_html=True)
        elif image_source == "📸 Take photo":
            st.markdown("""
            <div class="info-box">
                <h4>📸 Camera Ready</h4>
                <p>Use the camera button above to capture a receipt photo</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="info-box">
                <h4>🗂️ Sample Images Available</h4>
                <p>Select a sample receipt from the dropdown above</p>
            </div>
            """, unsafe_allow_html=True)
    
    # Display results if available (independent of current image selection)
    if (hasattr(st.session_state, 'ocr_llm_structured_data') and st.session_state.ocr_llm_structured_data) or \
       (hasattr(st.session_state, 'direct_llm_structured_data') and st.session_state.direct_llm_structured_data):
        
        st.markdown('<div class="section-header">📊 Analysis Results</div>', unsafe_allow_html=True)
        
        # Create tabs for different analysis methods
        if hasattr(st.session_state, 'ocr_llm_structured_data') and hasattr(st.session_state, 'direct_llm_structured_data'):
            # Both methods available - show both results and raw data
            tab1, tab2, tab3 = st.tabs(["🔍 OCR+LLM Results", "🤖 Direct LLM Results", "📄 Raw Data"])
            tab4 = tab3  # Raw data tab
        elif hasattr(st.session_state, 'ocr_llm_structured_data'):
            # Only OCR+LLM available
            tab1, tab4 = st.tabs(["🔍 OCR+LLM Results", "📄 Raw Data"])
            tab2 = tab3 = None
        else:
            # Only Direct LLM available  
            tab2, tab4 = st.tabs(["🤖 Direct LLM Results", "📄 Raw Data"])
            tab1 = tab3 = None
        
        # OCR+LLM Results Tab
        if tab1 and hasattr(st.session_state, 'ocr_llm_structured_data'):
            with tab1:
                st.markdown("### 🔍➡️🤖 OCR+LLM Analysis (Vision → Gemini)")
                
                if st.session_state.ocr_llm_structured_data:
                    structured = st.session_state.ocr_llm_structured_data
                    
                    # Store Information
                    st.markdown("#### 🏪 Store Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if structured.get('store_name'):
                            st.write(f"**Store Name:** {structured['store_name']}")
                        if structured.get('store_address'):
                            st.write(f"**Address:** {structured['store_address']}")
                        if structured.get('phone_number'):
                            st.write(f"**Phone:** {structured['phone_number']}")
                        if structured.get('cashier'):
                            st.write(f"**Cashier:** {structured['cashier']}")
                    
                    with col2:
                        if structured.get('date'):
                            st.write(f"**Date:** {structured['date']}")
                        if structured.get('time'):
                            st.write(f"**Time:** {structured['time']}")
                        if structured.get('receipt_number'):
                            st.write(f"**Receipt #:** {structured['receipt_number']}")
                    
                    # Items
                    if structured.get('items'):
                        st.markdown("#### 🛒 Items")
                        items_df_data = []
                        for item in structured['items']:
                            row = {'Item': item.get('name', 'N/A'), 'Price': item.get('price', 'N/A')}
                            if item.get('quantity'):
                                row['Quantity'] = item['quantity']
                            items_df_data.append(row)
                        st.dataframe(items_df_data, use_container_width=True)
                    
                    # Totals
                    st.markdown("#### 💰 Totals")
                    total_col1, total_col2, total_col3 = st.columns(3)
                    
                    with total_col1:
                        if structured.get('subtotal'):
                            st.metric("Subtotal", structured['subtotal'])
                    
                    with total_col2:
                        if structured.get('tax'):
                            st.metric("Tax", structured['tax'])
                    
                    with total_col3:
                        if structured.get('total'):
                            st.metric("Total", structured['total'])
                    
                    if structured.get('payment_method'):
                        st.write(f"**Payment Method:** {structured['payment_method']}")
                    
                    if structured.get('additional_info'):
                        st.markdown("#### ℹ️ Additional Information")
                        st.write(structured['additional_info'])
        
        # Direct LLM Results Tab
        if tab2 and hasattr(st.session_state, 'direct_llm_structured_data'):
            with tab2:
                st.markdown("### 🤖 Direct LLM Analysis (Gemini Direct)")
                
                if st.session_state.direct_llm_structured_data:
                    llm_data = st.session_state.direct_llm_structured_data
                    
                    # Store Information
                    st.markdown("#### 🏪 Store Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        if llm_data.get('store_name'):
                            st.write(f"**Store Name:** {llm_data['store_name']}")
                        if llm_data.get('store_address'):
                            st.write(f"**Address:** {llm_data['store_address']}")
                        if llm_data.get('phone_number'):
                            st.write(f"**Phone:** {llm_data['phone_number']}")
                        if llm_data.get('cashier'):
                            st.write(f"**Cashier:** {llm_data['cashier']}")
                    
                    with col2:
                        if llm_data.get('date'):
                            st.write(f"**Date:** {llm_data['date']}")
                        if llm_data.get('time'):
                            st.write(f"**Time:** {llm_data['time']}")
                        if llm_data.get('receipt_number'):
                            st.write(f"**Receipt #:** {llm_data['receipt_number']}")
                    
                    # Items
                    if llm_data.get('items'):
                        st.markdown("#### 🛒 Items")
                        items_df_data = []
                        for item in llm_data['items']:
                            row = {'Item': item.get('name', 'N/A'), 'Price': item.get('price', 'N/A')}
                            if item.get('quantity'):
                                row['Quantity'] = item['quantity']
                            items_df_data.append(row)
                        st.dataframe(items_df_data, use_container_width=True)
                    
                    # Totals
                    st.markdown("#### 💰 Totals")
                    total_col1, total_col2, total_col3 = st.columns(3)
                    
                    with total_col1:
                        if llm_data.get('subtotal'):
                            st.metric("Subtotal", llm_data['subtotal'])
                    
                    with total_col2:
                        if llm_data.get('tax'):
                            st.metric("Tax", llm_data['tax'])
                    
                    with total_col3:
                        if llm_data.get('total'):
                            st.metric("Total", llm_data['total'])
                    
                    if llm_data.get('payment_method'):
                        st.write(f"**Payment Method:** {llm_data['payment_method']}")
                    
                    if llm_data.get('additional_info'):
                        st.markdown("#### ℹ️ Additional Information")
                        st.write(llm_data['additional_info'])
        
        # Raw Data Tab
        if tab4:
            with tab4:
                st.markdown("### 📄 Raw Analysis Data")
                
                if hasattr(st.session_state, 'ocr_extracted_text'):
                    st.markdown("#### 🔍 OCR Extracted Text")
                    st.text_area(
                        "Complete extracted text:",
                        value=st.session_state.ocr_extracted_text,
                        height=150,
                        disabled=True
                    )
                
                if hasattr(st.session_state, 'ocr_llm_raw_response'):
                    st.markdown("#### 🔍➡️🤖 OCR+LLM Raw Response")
                    st.text_area(
                        "Complete OCR+LLM response:",
                        value=st.session_state.ocr_llm_raw_response,
                        height=150,
                        disabled=True
                    )
                
                if hasattr(st.session_state, 'direct_llm_raw_response'):
                    st.markdown("#### 🤖 Direct LLM Raw Response")
                    st.text_area(
                        "Complete Direct LLM response:",
                        value=st.session_state.direct_llm_raw_response,
                        height=150,
                        disabled=True
                    )

if __name__ == "__main__":
    main() 