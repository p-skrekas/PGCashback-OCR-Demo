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
                st.error(f"Credentials file not found: {credentials_path}")
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
                st.error(f"Credentials file not found: {credentials_path}")
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
        layout="wide"
    )
    
    # Authentication
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        # Login screen
        st.title("🔐 Login Required")
        st.markdown("Please enter your credentials to access the Receipt Scanner")
        
        with st.form("login_form"):
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown("### 👤 Authentication")
                username = st.text_input("Username", placeholder="Enter username")
                password = st.text_input("Password", type="password", placeholder="Enter password")
                
                login_button = st.form_submit_button("🚀 Login", use_container_width=True)
                
                if login_button:
                    if username == "reborrn" and password == "password":
                        st.session_state.authenticated = True
                        st.success("✅ Login successful! Redirecting...")
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
        
        # Add some styling
        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #666; font-size: 14px;'>
                🧾 Receipt Scanner: OCR+LLM vs Direct LLM Comparison Tool
            </div>
            """, 
            unsafe_allow_html=True
        )
        return
    
    # Logout button in sidebar
    with st.sidebar:
        st.markdown("### 👤 User Session")
        st.write("Logged in as: **reborrn**")
        if st.button("🚪 Logout"):
            st.session_state.authenticated = False
            # Clear all other session state variables
            for key in list(st.session_state.keys()):
                if key != 'authenticated':
                    del st.session_state[key]
            st.rerun()
    
    # App title
    st.title("🧾 Receipt Scanner: OCR+LLM vs Direct LLM Comparison")
    st.markdown("Compare traditional OCR→LLM pipeline with direct image analysis using Vertex AI")
    st.markdown("---")
    
    # Initialize clients
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔍 OCR + LLM Pipeline")
        vision_client = setup_google_vision()
        if vision_client:
            st.success("✅ Google Vision API connected")
        else:
            st.error("❌ Google Vision not available")
    
    with col2:
        st.subheader("🤖 Direct LLM Analysis")
        vertex_model = setup_vertex_ai()
        if vertex_model:
            st.success("✅ Vertex AI connected")
        else:
            st.error("❌ Vertex AI not available")
            with st.expander("🔑 Setup Vertex AI"):
                st.markdown("""
                **To use Vertex AI analysis:**
                1. Create a Google Cloud Project
                2. Enable the Vision API and Vertex AI API
                3. Create a service account and download the JSON key
                4. Place the key file in your app directory
                """)
    
    if not vision_client and not vertex_model:
        st.warning("⚠️ No analysis methods available. Please check your setup.")
        return
    
    # File upload section
    st.markdown("---")
    st.subheader("📷 Upload Receipt Image")
    
    # Option to choose between upload or sample images
    image_source = st.radio(
        "Choose image source:",
        ["Upload your own image", "Use sample images"],
        horizontal=True
    )
    
    uploaded_file = None
    image_bytes = None
    image = None
    
    if image_source == "Upload your own image":
        uploaded_file = st.file_uploader(
            "Choose a receipt image...",
            type=['png', 'jpg', 'jpeg'],
            help="Supported formats: PNG, JPG, JPEG"
        )
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image_bytes = uploaded_file.getvalue()
    
    else:  # Use sample images
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
        # Display the image
        try:
            # Create tabs for viewing options
            image_tab1, image_tab2 = st.tabs(["Original Image", "OCR Visualization"])
            
            with image_tab1:
                st.subheader("📷 Receipt Image")
                st.image(image, caption="Receipt Image", width=600)
                
                # Image info
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Filename", uploaded_file.name)
                with col2:
                    st.metric("File Size", f"{len(image_bytes) / 1024:.1f} KB")
                with col3:
                    st.metric("Dimensions", f"{image.size[0]} x {image.size[1]}")
                with col4:
                    st.metric("Format", image.format)
            
            # Analysis section
            st.markdown("---")
            st.subheader("🚀 Analysis Options")
            
            # Analysis buttons
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                if vision_client and vertex_model:
                    if st.button("🔍➡️🤖 Analyze with OCR+LLM", type="primary", use_container_width=True):
                        with st.spinner("Step 1: Extracting text with Google Vision OCR..."):
                            # Extract text using Google Vision
                            extracted_text, text_annotations = extract_text_with_vision(image_bytes, vision_client)
                            
                            if extracted_text:
                                # Create the visualization image with bounding boxes
                                if text_annotations:
                                    visualization_image = draw_bounding_boxes(image_bytes, text_annotations)
                                    if visualization_image:
                                        st.session_state.ocr_visualization_image = visualization_image
                                
                                with st.spinner("Step 2: Analyzing extracted text with Gemini..."):
                                    # Analyze the extracted text with LLM
                                    ocr_llm_structured_data, ocr_llm_raw_response = analyze_text_with_llm(extracted_text, vertex_model)
                                    
                                    if ocr_llm_structured_data or ocr_llm_raw_response:
                                        # Store OCR+LLM results in session state
                                        st.session_state.ocr_extracted_text = extracted_text
                                        st.session_state.ocr_llm_structured_data = ocr_llm_structured_data
                                        st.session_state.ocr_llm_raw_response = ocr_llm_raw_response
                                        
                                        st.success("✅ OCR+LLM processing completed!")
                                    else:
                                        st.error("❌ Failed to analyze extracted text with LLM.")
                            else:
                                st.error("❌ Failed to extract text with OCR.")
                else:
                    st.button("🔍➡️🤖 Analyze with OCR+LLM", disabled=True, use_container_width=True)
                    st.caption("Requires both Vision API and Vertex AI")
            
            with analysis_col2:
                if vertex_model:
                    if st.button("🤖 Analyze with Direct LLM", type="secondary", use_container_width=True):
                        with st.spinner("Processing image directly with Gemini..."):
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
                    st.button("🤖 Analyze with Direct LLM", disabled=True, use_container_width=True)
                    st.caption("Vertex AI not available")
            
            # Display OCR visualization in the second tab if available
            with image_tab2:
                st.subheader("🔍 OCR Text Detection Visualization")
                if hasattr(st.session_state, 'ocr_visualization_image') and st.session_state.ocr_visualization_image is not None:
                    st.image(st.session_state.ocr_visualization_image, caption="Receipt with OCR Text Detection", width=600)
                    st.info("🔎 Green boxes show detected text with labels")
                else:
                    st.info("👆 Click 'Analyze with OCR+LLM' to see text detection visualization")
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    # Display results if available (independent of current image selection)
    if (hasattr(st.session_state, 'ocr_llm_structured_data') and st.session_state.ocr_llm_structured_data) or \
       (hasattr(st.session_state, 'direct_llm_structured_data') and st.session_state.direct_llm_structured_data):
        
        st.markdown("---")
        st.subheader("📊 Analysis Results Comparison")
        
        # Create tabs for different analysis methods
        if hasattr(st.session_state, 'ocr_llm_structured_data') and hasattr(st.session_state, 'direct_llm_structured_data'):
            # Both methods available - show comparison
            tab1, tab2, tab3, tab4 = st.tabs(["📋 OCR+LLM Results", "🤖 Direct LLM Results", "⚖️ Comparison", "📄 Raw Data"])
        elif hasattr(st.session_state, 'ocr_llm_structured_data'):
            # Only OCR+LLM available
            tab1, tab4 = st.tabs(["📋 OCR+LLM Results", "📄 Raw Data"])
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
        
        # Comparison Tab
        if tab3 and hasattr(st.session_state, 'ocr_llm_structured_data') and hasattr(st.session_state, 'direct_llm_structured_data'):
            with tab3:
                st.markdown("### ⚖️ OCR+LLM vs Direct LLM Comparison")
                
                ocr_llm_data = st.session_state.ocr_llm_structured_data
                direct_llm_data = st.session_state.direct_llm_structured_data
                
                # Create comparison table
                comparison_data = {
                    'Field': [],
                    'OCR+LLM Result': [],
                    'Direct LLM Result': [],
                    'Match': []
                }
                
                fields_to_compare = [
                    ('store_name', 'Store Name'),
                    ('store_address', 'Store Address'),
                    ('phone_number', 'Phone Number'),
                    ('date', 'Date'),
                    ('time', 'Time'),
                    ('receipt_number', 'Receipt Number'),
                    ('subtotal', 'Subtotal'),
                    ('tax', 'Tax'),
                    ('total', 'Total'),
                    ('payment_method', 'Payment Method')
                ]
                
                for field_key, field_name in fields_to_compare:
                    ocr_value = ocr_llm_data.get(field_key) or "Not detected"
                    direct_value = direct_llm_data.get(field_key) or "Not detected"
                    
                    # Simple comparison (you could make this more sophisticated)
                    match = "✅" if str(ocr_value).strip().lower() == str(direct_value).strip().lower() else "❌"
                    if ocr_value == "Not detected" and direct_value == "Not detected":
                        match = "➖"
                    
                    comparison_data['Field'].append(field_name)
                    comparison_data['OCR+LLM Result'].append(str(ocr_value))
                    comparison_data['Direct LLM Result'].append(str(direct_value))
                    comparison_data['Match'].append(match)
                
                st.dataframe(comparison_data, use_container_width=True)
                
                # Items comparison
                st.markdown("#### 🛒 Items Comparison")
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**OCR+LLM Items:**")
                    if ocr_llm_data.get('items'):
                        for item in ocr_llm_data['items']:
                            item_text = f"- {item.get('name', 'N/A')}: {item.get('price', 'N/A')}"
                            if item.get('quantity'):
                                item_text += f" (Qty: {item['quantity']})"
                            st.write(item_text)
                    else:
                        st.write("No items detected")
                
                with col2:
                    st.write("**Direct LLM Items:**")
                    if direct_llm_data.get('items'):
                        for item in direct_llm_data['items']:
                            item_text = f"- {item.get('name', 'N/A')}: {item.get('price', 'N/A')}"
                            if item.get('quantity'):
                                item_text += f" (Qty: {item['quantity']})"
                            st.write(item_text)
                    else:
                        st.write("No items detected")
        
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
        
        # Download options
        st.markdown("---")
        st.subheader("💾 Download Results")
        
        download_cols = st.columns(4)
        
        with download_cols[0]:
            # Download OCR text
            if hasattr(st.session_state, 'ocr_extracted_text'):
                st.download_button(
                    label="📄 OCR Text",
                    data=st.session_state.ocr_extracted_text,
                    file_name="ocr_text.txt",
                    mime="text/plain"
                )
        
        with download_cols[1]:
            # Download OCR+LLM structured data
            if hasattr(st.session_state, 'ocr_llm_structured_data'):
                structured_json = json.dumps(st.session_state.ocr_llm_structured_data, indent=2)
                st.download_button(
                    label="🗂️ OCR+LLM Data",
                    data=structured_json,
                    file_name="ocr_llm_structured.json",
                    mime="application/json"
                )
        
        with download_cols[2]:
            # Download Direct LLM data
            if hasattr(st.session_state, 'direct_llm_structured_data'):
                llm_json = json.dumps(st.session_state.direct_llm_structured_data, indent=2)
                st.download_button(
                    label="🤖 Direct LLM Data",
                    data=llm_json,
                    file_name="direct_llm_structured.json",
                    mime="application/json"
                )
        
        with download_cols[3]:
            # Download visualization image if available
            if hasattr(st.session_state, 'ocr_visualization_image'):
                buf = io.BytesIO()
                st.session_state.ocr_visualization_image.save(buf, format="PNG")
                byte_im = buf.getvalue()
                
                st.download_button(
                    label="📸 OCR Visualization",
                    data=byte_im,
                    file_name="ocr_visualization.png",
                    mime="image/png"
                )
    
    else:
        # Show instructions when no image is selected
        if image_source == "Upload your own image":
            st.info("👆 Please upload a receipt image to get started")
        else:
            st.info("👆 Please select a sample receipt image to get started")
        
        # Add some helpful tips
        with st.expander("📝 Tips for best results"):
            st.markdown("""
            **For optimal analysis:**
            - Use clear, well-lit photos
            - Ensure the entire receipt is visible and flat
            - Avoid shadows, glare, and reflections
            - Take photos straight-on (not at an angle)
            - Use higher resolution images when possible
            - Ensure good contrast between text and background
            
            **Image sources:**
            - **Upload your own:** Use your own receipt images
            - **Sample images:** Select from pre-loaded examples in `receipts/images/`
            
            **Supported formats:** PNG, JPG, JPEG
            """)
        
        # Explain the features
        with st.expander("🔍 What does this app do?"):
            st.markdown("""
            **This Receipt Scanner compares two different approaches to AI-powered receipt analysis:**
            
            **🔍➡️🤖 OCR+LLM Pipeline (Traditional Approach):**
            - **Step 1**: Extract text using Google Vision API (OCR)
            - **Step 2**: Analyze extracted text with Gemini Flash 2.0 (LLM)
            - Combines reliable text extraction with intelligent analysis
            - Good for clear, well-formatted receipts
            - Shows intermediate OCR text for transparency
            
            **🤖 Direct LLM Analysis (Modern Approach):**
            - **Single Step**: Gemini Flash 2.0 analyzes the image directly
            - No intermediate OCR step - understands images natively
            - Better at handling poor quality, rotated, or unusual layouts
            - Can see context that pure text extraction might miss
            - More end-to-end but less transparent
            
            **⚖️ Comparison Features:**
            - Side-by-side structured results
            - Field-by-field accuracy comparison with match indicators
            - Performance insights for different receipt types
            - Raw data inspection for both approaches
            - Visual OCR detection overlay (pipeline method only)
            
            **📊 What you'll see extracted:**
            - **Store Information**: Name, address, phone number
            - **Transaction Details**: Date, time, receipt number
            - **Itemized List**: Products, prices, quantities
            - **Financial Summary**: Subtotal, tax, total amounts
            - **Payment Method**: Cash, card, digital payments
            - **Additional Insights**: Cashier names, promotions, discounts
            
            **🎯 Use Cases:**
            - **Pipeline approach**: When you need transparency and auditability
            - **Direct approach**: When handling diverse or poor-quality images
            - **Comparison**: To understand which method works better for your data
            
            This comparison helps you choose the optimal approach for your specific receipt processing needs!
            """)
        
        # Setup instructions
        with st.expander("⚙️ Setup Instructions"):
            st.markdown("""
            **Google Cloud Setup (required for both methods):**
            1. Create a Google Cloud Project
            2. Enable the following APIs:
               - Cloud Vision API (for OCR+LLM pipeline)
               - Vertex AI API (for both methods)
            3. Create a service account and download the JSON key
            4. Place the key file in your app directory
            5. Ensure your service account has the following roles:
               - Cloud Vision API User (for OCR functionality)
               - Vertex AI User (for LLM functionality)
            
            **Note:** Both analysis methods use the same GCP credentials - no separate API keys needed!
            
            **Cost Considerations:**
            - OCR+LLM: Vision API calls + Vertex AI text processing
            - Direct LLM: Vertex AI image processing (typically higher cost per request)
            """)

if __name__ == "__main__":
    main() 