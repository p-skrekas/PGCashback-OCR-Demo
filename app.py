# -*- coding: utf-8 -*-
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
            with open(credentials_path, 'r', encoding='utf-8') as f:
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

def draw_unicode_text_on_image(img, text, position, font_size=20, color=(255, 0, 0)):
    """Draw Unicode text (including Greek characters) on image using PIL"""
    try:
        # Convert OpenCV image to PIL
        img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(img_pil)
        
        # Try to use a font that supports Unicode
        try:
            # Try to load a system font that supports Greek characters
            font = ImageFont.truetype("/System/Library/Fonts/Arial.ttf", font_size)
        except:
            try:
                # Fallback for different systems
                font = ImageFont.truetype("arial.ttf", font_size)
            except:
                try:
                    # Another fallback
                    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
                except:
                    # Use default font as last resort
                    font = ImageFont.load_default()
        
        # Draw the text
        draw.text(position, text, font=font, fill=color)
        
        # Convert back to OpenCV format
        img_cv = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
        return img_cv
        
    except Exception as e:
        print(f"Error drawing Unicode text: {e}")
        # Fallback to original image if text drawing fails
        return img

def draw_bounding_boxes(image_bytes, text_annotations):
    """Draw bounding boxes around detected text in the image with Unicode support"""
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
        
        # Convert to PIL Image for Unicode text support
        img_pil = Image.fromarray(img)
        draw = ImageDraw.Draw(img_pil)
        
        # Try to load a Unicode-compatible font
        try:
            # Try different font paths for different systems
            font_paths = [
                "/System/Library/Fonts/Arial.ttf",  # macOS
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",  # Linux
                "C:/Windows/Fonts/arial.ttf",  # Windows
                "arial.ttf"  # Generic
            ]
            
            font = None
            font_size = 12 if scale_factor < 1.0 else 16
            
            for font_path in font_paths:
                try:
                    font = ImageFont.truetype(font_path, font_size)
                    break
                except:
                    continue
            
            if font is None:
                font = ImageFont.load_default()
                
        except:
            font = ImageFont.load_default()
        
        # Draw bounding boxes (scaled if necessary)
        for text in text_annotations:
            points = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            
            # Scale the coordinates if image was resized
            if scale_factor != 1.0:
                points = [(int(x * scale_factor), int(y * scale_factor)) for x, y in points]
            
            # Create a rectangle
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Draw the rectangle using PIL
            draw.rectangle([(x1, y1), (x2, y2)], outline=(0, 255, 0), width=2)
            
            # Add text description with Unicode support
            # Display a truncated version if the text is too long
            display_text = text.description[:12] + "..." if len(text.description) > 12 else text.description
            
            # Ensure the text is properly encoded as UTF-8
            try:
                display_text = display_text.encode('utf-8').decode('utf-8')
            except:
                display_text = "???"  # Fallback for problematic characters
            
            # Draw text with Unicode support
            text_position = (x1, max(0, y1 - 20))
            draw.text(text_position, display_text, font=font, fill=(255, 0, 0))
        
        # Convert back to PIL Image and return
        return img_pil
    
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
        # Ensure the extracted text is properly encoded as UTF-8
        if extracted_text:
            try:
                extracted_text = extracted_text.encode('utf-8').decode('utf-8')
            except:
                extracted_text = extracted_text  # Keep original if encoding fails
        
        # Create the prompt for text analysis
        prompt = f"""
        Analyze this receipt text that was extracted using OCR and provide a clear, structured analysis. 
        Be as accurate as possible and only include information that is clearly present in the text.
        
        Receipt Text:
        {extracted_text}
        
        Please analyze and extract the following information in a clear, readable format:
        
        STORE INFORMATION:
        - Store name
        - Store address  
        - Phone number
        - Cashier name (if visible)
        
        TRANSACTION DETAILS:
        - Date of transaction
        - Time of transaction
        - Receipt or transaction number
        
        ITEMS PURCHASED:
        - List each item with its price and quantity (if specified)
        
        FINANCIAL SUMMARY:
        - Subtotal amount
        - Tax amount
        - Total amount
        - Payment method used
        
        ADDITIONAL INFORMATION:
        - Any discounts, promotions, or other relevant details
        
        Format your response in a clear, organized manner. If any information is not clearly visible in the receipt text, indicate that it is "Not clearly visible" or "Not specified".
        Please preserve any non-English characters (like Greek letters) exactly as they appear.
        Response only with the information and do not provide any comments,
        """
        
        # Generate content with the text prompt
        response = model.generate_content(prompt)
        
        # Return only the raw response text with proper encoding
        response_text = response.text.strip() if response.text else ""
        
        # Ensure proper UTF-8 encoding of the response
        try:
            response_text = response_text.encode('utf-8').decode('utf-8')
        except:
            pass  # Keep original if encoding fails
            
        return response_text
            
    except Exception as e:
        st.error(f"Error analyzing text with LLM: {str(e)}")
        return None

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
        Analyze this receipt image directly and provide a clear, structured analysis. 
        Be as accurate as possible and only include information that is clearly visible in the receipt.
        
        Please analyze and extract the following information in a clear, readable format:
        
        STORE INFORMATION:
        - Store name
        - Store address
        - Phone number
        - Cashier name (if visible)
        
        TRANSACTION DETAILS:
        - Date of transaction
        - Time of transaction
        - Receipt or transaction number
        
        ITEMS PURCHASED:
        - List each item with its price and quantity (if specified)
        
        FINANCIAL SUMMARY:
        - Subtotal amount
        - Tax amount
        - Total amount
        - Payment method used
        
        ADDITIONAL INFORMATION:
        - Any discounts, promotions, or other relevant details
        
        Format your response in a clear, organized manner. If any information is not clearly visible or readable in the receipt, indicate that it is "Not clearly visible" or "Not specified".
        Please preserve any non-English characters (like Greek letters) exactly as they appear.
        """
        
        # Generate content with the image and prompt
        response = model.generate_content([prompt, image_part])
        
        # Return only the raw response text with proper encoding
        response_text = response.text.strip() if response.text else ""
        
        # Ensure proper UTF-8 encoding of the response
        try:
            response_text = response_text.encode('utf-8').decode('utf-8')
        except:
            pass  # Keep original if encoding fails
            
        return response_text
            
    except Exception as e:
        st.error(f"Error analyzing receipt with Direct LLM: {str(e)}")
        return None

def detect_receipt_boundaries(image_bytes, client):
    """Detect receipt boundaries using Google Vision API"""
    try:
        # Create Vision Image object
        image = vision.Image(content=image_bytes)
        
        # Perform document text detection to get overall document boundaries
        response = client.document_text_detection(image=image)
        
        if response.error.message:
            raise Exception(f'{response.error.message}')
        
        # Get the full text annotation which contains page-level information
        document = response.full_text_annotation
        
        if document and document.pages:
            # Get the first page
            page = document.pages[0]
            
            # Get page dimensions and bounds
            page_width = page.width
            page_height = page.height
            
            # Find the bounding box that encompasses most text blocks
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0
            
            for block in page.blocks:
                for vertex in block.bounding_box.vertices:
                    min_x = min(min_x, vertex.x)
                    min_y = min(min_y, vertex.y)
                    max_x = max(max_x, vertex.x)
                    max_y = max(max_y, vertex.y)
            
            # Add some padding around the detected text area
            padding_x = int((max_x - min_x) * 0.05)  # 5% padding
            padding_y = int((max_y - min_y) * 0.05)
            
            # Ensure bounds are within image dimensions
            crop_left = max(0, min_x - padding_x)
            crop_top = max(0, min_y - padding_y)
            crop_right = min(page_width, max_x + padding_x)
            crop_bottom = min(page_height, max_y + padding_y)
            
            # Convert to percentages for consistency with manual cropping
            left_percent = int((crop_left / page_width) * 100)
            top_percent = int((crop_top / page_height) * 100)
            right_percent = int((crop_right / page_width) * 100)
            bottom_percent = int((crop_bottom / page_height) * 100)
            
            return {
                'left': left_percent,
                'top': top_percent,
                'right': right_percent,
                'bottom': bottom_percent,
                'confidence': 'high' if len(page.blocks) > 3 else 'medium'
            }
        
        # Fallback: use regular text detection
        text_response = client.text_detection(image=image)
        texts = text_response.text_annotations
        
        if texts and len(texts) > 1:
            # Skip the first annotation (full text) and process individual text blocks
            min_x, min_y = float('inf'), float('inf')
            max_x, max_y = 0, 0
            
            for text in texts[1:]:  # Skip first element which is full text
                for vertex in text.bounding_poly.vertices:
                    min_x = min(min_x, vertex.x)
                    min_y = min(min_y, vertex.y)
                    max_x = max(max_x, vertex.x)
                    max_y = max(max_y, vertex.y)
            
            # Estimate image dimensions from the text bounds
            img_width = max_x * 1.2  # Estimate with some margin
            img_height = max_y * 1.2
            
            # Add padding
            padding_x = int((max_x - min_x) * 0.1)
            padding_y = int((max_y - min_y) * 0.1)
            
            crop_left = max(0, min_x - padding_x)
            crop_top = max(0, min_y - padding_y)
            crop_right = min(img_width, max_x + padding_x)
            crop_bottom = min(img_height, max_y + padding_y)
            
            # Convert to percentages
            left_percent = max(0, int((crop_left / img_width) * 100))
            top_percent = max(0, int((crop_top / img_height) * 100))
            right_percent = min(100, int((crop_right / img_width) * 100))
            bottom_percent = min(100, int((crop_bottom / img_height) * 100))
            
            return {
                'left': left_percent,
                'top': top_percent,
                'right': right_percent,
                'bottom': bottom_percent,
                'confidence': 'medium'
            }
        
        return None
        
    except Exception as e:
        st.warning(f"Could not detect receipt boundaries: {str(e)}")
        return None

def visualize_detected_boundaries(image, boundaries):
    """Visualize detected boundaries on the image"""
    try:
        # Create a copy of the image for drawing
        viz_image = image.copy()
        draw = ImageDraw.Draw(viz_image)
        
        # Get image dimensions
        img_width, img_height = viz_image.size
        
        # Convert percentage boundaries to pixel coordinates
        left = int(img_width * boundaries['left'] / 100)
        top = int(img_height * boundaries['top'] / 100)
        right = int(img_width * boundaries['right'] / 100)
        bottom = int(img_height * boundaries['bottom'] / 100)
        
        # Draw boundary rectangle
        outline_color = "red" if boundaries['confidence'] == 'high' else "orange"
        line_width = 3
        
        # Draw the boundary rectangle
        draw.rectangle(
            [(left, top), (right, bottom)],
            outline=outline_color,
            width=line_width
        )
        
        # Add corner markers for better visibility
        marker_size = 10
        corners = [
            (left, top), (right, top),
            (left, bottom), (right, bottom)
        ]
        
        for corner_x, corner_y in corners:
            draw.rectangle(
                [(corner_x - marker_size, corner_y - marker_size),
                 (corner_x + marker_size, corner_y + marker_size)],
                fill=outline_color
            )
        
        return viz_image
        
    except Exception as e:
        st.warning(f"Could not visualize boundaries: {str(e)}")
        return image

def detect_document_edges_advanced(image_bytes):
    """Advanced document edge detection using OpenCV Canny algorithm"""
    try:
        import cv2
        import numpy as np
        
        # Convert bytes to OpenCV image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            return None
            
        original_height, original_width = img.shape[:2]
        
        # Convert to grayscale
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # 1. Noise Reduction with Gaussian Blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # 2. Apply adaptive threshold for better edge detection
        adaptive_thresh = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2
        )
        
        # 3. Canny Edge Detection with optimized parameters for documents
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150, apertureSize=3)
        
        # 4. Morphological operations to close gaps in edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # 5. Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return None
            
        # 6. Filter contours by area (remove very small contours)
        min_area = (original_width * original_height) * 0.1  # At least 10% of image
        large_contours = [c for c in contours if cv2.contourArea(c) > min_area]
        
        if not large_contours:
            return None
            
        # 7. Find the contour with the most rectangular shape
        best_contour = None
        best_score = 0
        
        for contour in large_contours:
            # Approximate contour to polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            
            # Score based on how close to rectangle (4 vertices) and area
            area = cv2.contourArea(contour)
            vertices_score = 4.0 / max(len(approx), 4)  # Prefer 4 vertices
            area_score = area / (original_width * original_height)  # Prefer larger areas
            
            score = vertices_score * area_score
            
            if score > best_score and len(approx) >= 4:
                best_score = score
                best_contour = approx
        
        if best_contour is not None:
            # Convert contour points to bounding box percentages
            x_coords = [point[0][0] for point in best_contour]
            y_coords = [point[0][1] for point in best_contour]
            
            left = min(x_coords)
            right = max(x_coords)
            top = min(y_coords)
            bottom = max(y_coords)
            
            # Add small padding
            padding_x = int((right - left) * 0.02)
            padding_y = int((bottom - top) * 0.02)
            
            left = max(0, left - padding_x)
            right = min(original_width, right + padding_x)
            top = max(0, top - padding_y)
            bottom = min(original_height, bottom + padding_y)
            
            # Convert to percentages
            left_percent = int((left / original_width) * 100)
            top_percent = int((top / original_height) * 100)
            right_percent = int((right / original_width) * 100)
            bottom_percent = int((bottom / original_height) * 100)
            
            return {
                'left': left_percent,
                'top': top_percent,
                'right': right_percent,
                'bottom': bottom_percent,
                'confidence': 'high' if best_score > 0.5 else 'medium',
                'method': 'canny_edge_detection'
            }
        
        return None
        
    except Exception as e:
        st.warning(f"Advanced edge detection failed: {str(e)}")
        return None

def main():
    # Page configuration with Unicode support
    st.set_page_config(
        page_title="Receipt Scanner: OCR+LLM vs Direct LLM",
        page_icon="🧾",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Ensure UTF-8 encoding for the entire app
    import locale
    try:
        locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')
    except locale.Error:
        try:
            locale.setlocale(locale.LC_ALL, 'C.UTF-8')
        except locale.Error:
            pass  # Keep default locale if UTF-8 is not available
    

    
    # Sidebar info
    with st.sidebar:
        st.markdown("### 📊 Session Info")
        if hasattr(st.session_state, 'ocr_llm_response') or hasattr(st.session_state, 'direct_llm_response'):
            st.success("✅ Analysis data available")
        else:
            st.info("🔄 No analysis performed yet")
        
        st.markdown("---")
        st.markdown("### 🔧 About")
        st.markdown("Receipt analysis using OCR+LLM pipeline vs Direct LLM comparison.")
        st.markdown("Upload an image to get started!")
    
    # Simple main header
    st.title("Receipt Scanner")
    st.subheader("OCR+LLM vs Direct LLM")
    


    
    # Initialize clients (moved here since system status section was commented out)
    vision_client = setup_google_vision()
    vertex_model = setup_vertex_ai()
    
    # Enhanced image source selection
    st.markdown("Choose your preferred image input method:")
    image_source = st.radio(
        "",
        ["📁 Upload from device", "🗂️ Sample images"],
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
            # Image display
            st.header("📊 Image Analysis")
            
            # Create tabs for viewing options
            image_tab1, image_tab2 = st.tabs(["🖼️ Original Image", "🔍 OCR Visualization"])
            
            with image_tab1:
                st.image(image, caption="Receipt Image", width=600)
                
                # Image metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("📄 Filename", uploaded_file.name)
                with col2:
                    st.metric("💾 File Size", f"{len(image_bytes) / 1024:.1f} KB")
                with col3:
                    st.metric("📐 Dimensions", f"{image.size[0]} × {image.size[1]}")
                with col4:
                    st.metric("🎨 Format", image.format)
            
            # Analysis section
            st.header("🚀 AI Analysis")
            st.write("Choose your analysis method to process the receipt:")
            
            # Professional analysis buttons
            analysis_col1, analysis_col2 = st.columns(2)
            
            with analysis_col1:
                if vision_client and vertex_model:
                    if st.button("OCR + LLM Pipeline", type="primary", use_container_width=True):
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
                                    ocr_llm_response = analyze_text_with_llm(extracted_text, vertex_model)
                                    
                                    if ocr_llm_response:
                                        # Store OCR+LLM results in session state
                                        st.session_state.ocr_extracted_text = extracted_text
                                        st.session_state.ocr_llm_response = ocr_llm_response
                                        
                                        st.success("OCR+LLM pipeline analysis completed!")
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
                            direct_llm_response = analyze_receipt_with_llm(image_bytes, vertex_model)
                            
                            if direct_llm_response:
                                # Store Direct LLM results in session state
                                st.session_state.direct_llm_response = direct_llm_response
                                
                                st.success("✅ Direct LLM analysis completed!")
                            else:
                                st.error("❌ Failed to analyze with Direct LLM.")
                else:
                    st.button("🤖 Direct LLM Analysis", disabled=True, use_container_width=True)
                    st.caption("⚠️ Vertex AI not available")
            
            # Display OCR visualization in the second tab if available
            with image_tab2:
                st.subheader("🔍 OCR Text Detection")
                if hasattr(st.session_state, 'ocr_visualization_image') and st.session_state.ocr_visualization_image is not None:
                    st.image(st.session_state.ocr_visualization_image, caption="Receipt with OCR Text Detection", width=600)
                    st.info("🔎 Green boxes show detected text regions with labels")
                else:
                    st.info("👆 Click 'OCR + LLM Pipeline' to see text detection visualization")
        
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.info("Please try uploading a different image or check the file format.")
    
    else:

        
        if image_source == "📁 Upload from device":
            st.info("📁 **Upload Receipt Image** - Select a receipt image from your device to begin analysis")
        else:
            st.info("🗂️ **Sample Images Available** - Select a sample receipt from the dropdown above")
    
    # Display results if available (independent of current image selection)
    if (hasattr(st.session_state, 'ocr_llm_response') and st.session_state.ocr_llm_response) or \
       (hasattr(st.session_state, 'direct_llm_response') and st.session_state.direct_llm_response):
        
        st.header("📊 Analysis Results")
        
        # Create tabs for different analysis methods
        if hasattr(st.session_state, 'ocr_llm_response') and hasattr(st.session_state, 'direct_llm_response'):
            # Both methods available - show both results and extracted text
            tab1, tab2, tab3 = st.tabs(["🔍 OCR+LLM Results", "🤖 Direct LLM Results", "📄 Raw OCR Text"])
            tab4 = tab3  # Raw data tab
        elif hasattr(st.session_state, 'ocr_llm_response'):
            # Only OCR+LLM available
            tab1, tab4 = st.tabs(["🔍 OCR+LLM Results", "📄 Raw OCR Text"])
            tab2 = tab3 = None
        else:
            # Only Direct LLM available  
            tab2, tab4 = st.tabs(["🤖 Direct LLM Results", "📄 Analysis Data"])
            tab1 = tab3 = None
        
        # OCR+LLM Results Tab
        if tab1 and hasattr(st.session_state, 'ocr_llm_response'):
            with tab1:
                st.markdown("### 📊 Comparison: OCR+LLM vs Direct LLM")
                st.markdown("*Side-by-side comparison of both analysis methods*")
                
                # Display OCR+LLM and Direct LLM outputs side by side
                col1, col2 = st.columns(2)
                
                with col1:
                    st.markdown("#### 🔍➡️🤖 OCR+LLM Analysis")
                    if st.session_state.ocr_llm_response:
                        try:
                            # Ensure proper UTF-8 encoding for display
                            display_text = st.session_state.ocr_llm_response.encode('utf-8').decode('utf-8')
                            st.text_area(
                                "Analysis from OCR → LLM pipeline:",
                                value=display_text,
                                height=400,
                                disabled=True,
                                key="ocr_llm_output_display"
                            )
                        except:
                            # Fallback to original text if encoding fails
                            st.text_area(
                                "Analysis from OCR → LLM pipeline:",
                                value=st.session_state.ocr_llm_response,
                                height=400,
                                disabled=True,
                                key="ocr_llm_output_display_fallback"
                            )
                    else:
                        st.info("OCR+LLM analysis not available")
                
                with col2:
                    st.markdown("#### 🤖 Direct LLM Analysis")
                    if hasattr(st.session_state, 'direct_llm_response') and st.session_state.direct_llm_response:
                        try:
                            # Ensure proper UTF-8 encoding for display
                            display_text = st.session_state.direct_llm_response.encode('utf-8').decode('utf-8')
                            st.text_area(
                                "Analysis from direct image → LLM:",
                                value=display_text,
                                height=400,
                                disabled=True,
                                key="direct_llm_output_display"
                            )
                        except:
                            # Fallback to original text if encoding fails
                            st.text_area(
                                "Analysis from direct image → LLM:",
                                value=st.session_state.direct_llm_response,
                                height=400,
                                disabled=True,
                                key="direct_llm_output_display_fallback"
                            )
                    else:
                        st.info("Direct LLM analysis not available")
                        st.caption("Run 'Direct LLM Analysis' to compare both methods")
        
        # Direct LLM Results Tab
        if tab2 and hasattr(st.session_state, 'direct_llm_response'):
            with tab2:
                st.markdown("### 🤖 Direct LLM Analysis (Gemini Direct)")
                st.markdown("*Analysis based on direct image processing by AI*")
                
                if st.session_state.direct_llm_response:
                    # Display the raw LLM response in a formatted container with Unicode support
                    st.markdown("#### 📋 Analysis Results")
                    try:
                        # Ensure proper UTF-8 encoding for display
                        display_text = st.session_state.direct_llm_response.encode('utf-8').decode('utf-8')
                        st.markdown(f"```\n{display_text}\n```")
                    except:
                        # Fallback to original text if encoding fails
                        st.markdown(f"```\n{st.session_state.direct_llm_response}\n```")
        
        # Raw Data Tab
        if tab4:
            with tab4:
                st.markdown("### 📄 Raw Analysis Data")
                
                if hasattr(st.session_state, 'ocr_extracted_text'):
                    st.markdown("#### 🔍 OCR Extracted Text")
                    st.markdown("*Raw text extracted from the image using Google Vision OCR*")
                    st.text_area(
                        "Complete extracted text:",
                        value=st.session_state.ocr_extracted_text,
                        height=200,
                        disabled=True
                    )
                
                if hasattr(st.session_state, 'ocr_llm_response'):
                    st.markdown("#### 🔍➡️🤖 OCR+LLM Analysis")
                    st.markdown("*AI analysis of the OCR-extracted text*")
                    st.text_area(
                        "Complete OCR+LLM analysis:",
                        value=st.session_state.ocr_llm_response,
                        height=200,
                        disabled=True
                    )
                
                if hasattr(st.session_state, 'direct_llm_response'):
                    st.markdown("#### 🤖 Direct LLM Analysis")
                    st.markdown("*AI analysis directly from the image*")
                    st.text_area(
                        "Complete Direct LLM analysis:",
                        value=st.session_state.direct_llm_response,
                        height=200,
                        disabled=True
                    )

if __name__ == "__main__":
    main() 