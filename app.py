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

# Import our template system
import sys
sys.path.append('.')
from templates.template_model import TemplateManager, ReceiptTemplate

def setup_google_vision():
    """Initialize Google Vision client with specific credentials"""
    try:
        # Load credentials from the JSON file
        credentials_path = "pg-cashback-1ff07e84ca4f.json"
        
        if not os.path.exists(credentials_path):
            st.error(f"Credentials file not found: {credentials_path}")
            return None
        
        # Create credentials from the service account file
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        
        # Initialize the Vision client with specific credentials
        client = vision.ImageAnnotatorClient(credentials=credentials)
        return client
        
    except Exception as e:
        st.error(f"Error initializing Google Vision client: {str(e)}")
        st.info("Please ensure your Google Cloud credentials file is properly configured.")
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
        
        # Draw bounding boxes
        for text in text_annotations:
            points = [(vertex.x, vertex.y) for vertex in text.bounding_poly.vertices]
            
            # Create a rectangle (OpenCV requires integers for rectangle points)
            x_coords = [p[0] for p in points]
            y_coords = [p[1] for p in points]
            
            x1, y1 = min(x_coords), min(y_coords)
            x2, y2 = max(x_coords), max(y_coords)
            
            # Draw the rectangle
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            
            # Add text description
            # Display a truncated version if the text is too long
            display_text = text.description[:15] + "..." if len(text.description) > 15 else text.description
            cv2.putText(img, display_text, (int(x1), int(y1) - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Convert back to PIL Image
        return Image.fromarray(img)
    
    except Exception as e:
        st.error(f"Error drawing bounding boxes: {str(e)}")
        return None

def extract_structured_data_with_templates(text, template_manager):
    """Extract structured data using vendor templates first, fallback to generic parsing"""
    
    # Try template-based parsing first
    template_result = template_manager.parse_receipt(text)
    
    if template_result.get('template_match'):
        # Template found and matched
        return template_result
    
    # Fallback to generic parsing if no template matches
    return extract_structured_data_generic(text)

def extract_structured_data_generic(text):
    """Generic structured data extraction (original logic)"""
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
        'payment_method': None,
        'template_match': False
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

def main():
    # Page configuration
    st.set_page_config(
        page_title="Receipt Scanner with OCR",
        page_icon="🧾",
        layout="centered"
    )
    
    # App title
    st.title("🧾 Receipt Scanner with Vendor Templates")
    st.markdown("---")
    
    # Initialize template manager
    @st.cache_resource
    def load_template_manager():
        return TemplateManager()
    
    template_manager = load_template_manager()
    
    # Initialize Google Vision client
    vision_client = setup_google_vision()
    
    if not vision_client:
        st.warning("⚠️ Google Vision client not initialized. Please check your setup.")
        with st.expander("📋 Setup Instructions"):
            st.markdown("""
            **Current Setup:**
            - Using credentials from: `pg-cashback-1ff07e84ca4f.json`
            - Project ID: `pg-cashback`
            
            **Make sure:**
            1. The credentials file is in the same directory as this app
            2. The Google Cloud Vision API is enabled for your project
            3. The service account has the necessary permissions
            
            **If you're still having issues:**
            - Check that the Vision API is enabled in your Google Cloud Console
            - Verify the service account has 'Cloud Vision API User' role
            """)
        return
    
    # Show connection status and templates info
    col1, col2 = st.columns(2)
    with col1:
        st.success("✅ Connected to Google Vision API (Project: pg-cashback)")
    with col2:
        templates = template_manager.get_all_templates()
        st.info(f"📋 {len(templates)} vendor templates loaded")
    
    # Display available templates
    if templates:
        with st.expander("🏪 Available Vendor Templates"):
            template_names = [t['name'] for t in templates]
            st.write("**Supported Vendors:** " + ", ".join(template_names))
            st.caption("Receipts from these vendors will be parsed with enhanced accuracy using specialized templates.")
    
    # File upload section
    st.subheader("Upload Receipt Image")
    uploaded_file = st.file_uploader(
        "Choose a receipt image...",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display the uploaded image
        try:
            # Open and display the image
            image = Image.open(uploaded_file)
            
            # Get image bytes for OCR
            image_bytes = uploaded_file.getvalue()
            
            # Create tabs for viewing options
            image_tab1, image_tab2 = st.tabs(["Original Image", "OCR Visualization"])
            
            with image_tab1:
                st.subheader("📷 Original Receipt")
                st.image(image, caption="Receipt Image", use_container_width=True)
                
                # Image info
                st.write(f"**Filename:** {uploaded_file.name}")
                st.write(f"**File size:** {len(image_bytes) / 1024:.1f} KB")
                st.write(f"**Image dimensions:** {image.size[0]} x {image.size[1]} pixels")
                st.write(f"**Image format:** {image.format}")
            
            # Add OCR Processing section
            st.subheader("🔍 OCR Processing")
            
            # Add processing button
            if st.button("🚀 Analyze Receipt", type="primary"):
                with st.spinner("Processing with Google Vision OCR..."):
                    # Extract text using Google Vision
                    extracted_text, text_annotations = extract_text_with_vision(image_bytes, vision_client)
                    
                    if extracted_text:
                        # Store results in session state
                        st.session_state.extracted_text = extracted_text
                        st.session_state.text_annotations = text_annotations
                        st.session_state.analysis = analyze_receipt_text(extracted_text)
                        
                        # Use template-based parsing
                        with st.spinner("Applying vendor templates..."):
                            st.session_state.structured_data = extract_structured_data_with_templates(
                                extracted_text, template_manager
                            )
                        
                        # Create the visualization image with bounding boxes
                        if text_annotations:
                            with st.spinner("Creating visualization..."):
                                visualization_image = draw_bounding_boxes(image_bytes, text_annotations)
                                if visualization_image:
                                    st.session_state.visualization_image = visualization_image
                        
                        st.success("✅ OCR processing completed!")
                    else:
                        st.error("❌ Failed to extract text from the image.")
            
            # Display visualization in the second tab if available
            with image_tab2:
                st.subheader("🔍 OCR Visualization")
                if hasattr(st.session_state, 'visualization_image') and st.session_state.visualization_image is not None:
                    st.image(st.session_state.visualization_image, caption="Receipt with OCR Text Detection", use_container_width=True)
                    st.info("🔎 Green boxes show detected text with labels")
                else:
                    st.info("👆 Click 'Analyze Receipt' to see text detection visualization")
            
            # Display results if available
            if hasattr(st.session_state, 'extracted_text') and st.session_state.extracted_text:
                st.markdown("---")
                
                # Show template match status
                if hasattr(st.session_state, 'structured_data') and st.session_state.structured_data:
                    if st.session_state.structured_data.get('template_match'):
                        vendor_name = st.session_state.structured_data.get('vendor', 'Unknown')
                        st.success(f"🎯 **Template Match Found:** {vendor_name} - Enhanced parsing applied!")
                    else:
                        st.info("ℹ️ No vendor template matched - using generic parsing")
                
                st.subheader("📄 Extracted Text")
                
                # Create tabs for different views
                tab1, tab2, tab3, tab4 = st.tabs(["Raw Text", "Analysis", "Structured Data", "Template Management"])
                
                with tab1:
                    st.text_area(
                        "Complete extracted text:",
                        value=st.session_state.extracted_text,
                        height=200,
                        disabled=True
                    )
                
                with tab2:
                    if st.session_state.analysis:
                        analysis = st.session_state.analysis
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.metric("Total Lines", analysis['total_lines'])
                        
                        with col2:
                            st.metric("Potential Amounts", len(analysis['potential_amounts']))
                        
                        with col3:
                            st.metric("Potential Dates", len(analysis['potential_dates']))
                        
                        if analysis['potential_amounts']:
                            st.write("**💰 Detected Amounts:**")
                            for amount in analysis['potential_amounts']:
                                st.write(f"- {amount}")
                        
                        if analysis['potential_dates']:
                            st.write("**📅 Detected Dates:**")
                            for date in analysis['potential_dates']:
                                st.write(f"- {date}")
                        
                        if analysis['potential_store_info']:
                            st.write("**🏪 Store Information (top lines):**")
                            for i, line in enumerate(analysis['potential_store_info'], 1):
                                st.write(f"{i}. {line}")
                
                with tab3:
                    if hasattr(st.session_state, 'structured_data') and st.session_state.structured_data:
                        structured = st.session_state.structured_data
                        
                        # Template match indicator
                        if structured.get('template_match'):
                            st.success(f"✨ **Enhanced parsing using {structured.get('vendor', 'vendor')} template**")
                        else:
                            st.info("📝 **Using generic parsing (no template match)**")
                        
                        st.markdown("### 🏪 Store Information")
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            if structured.get('store_name'):
                                st.write(f"**Store Name:** {structured['store_name']}")
                            if structured.get('store_address'):
                                st.write(f"**Address:** {structured['store_address']}")
                            if structured.get('phone_number'):
                                st.write(f"**Phone:** {structured['phone_number']}")
                            # Show additional vendor-specific fields
                            if structured.get('store_number'):
                                st.write(f"**Store Number:** {structured['store_number']}")
                            if structured.get('membership_number'):
                                st.write(f"**Membership #:** {structured['membership_number']}")
                        
                        with col2:
                            if structured.get('date'):
                                st.write(f"**Date:** {structured['date']}")
                            if structured.get('time'):
                                st.write(f"**Time:** {structured['time']}")
                            if structured.get('receipt_number'):
                                st.write(f"**Receipt #:** {structured['receipt_number']}")
                            if structured.get('vendor_id'):
                                st.write(f"**Template Used:** {structured['vendor_id']}")
                        
                        st.markdown("---")
                        
                        # Items
                        if structured.get('items'):
                            st.markdown("### 🛒 Items")
                            items_df_data = []
                            for item in structured['items']:
                                row_data = {
                                    'Item': item['name'],
                                    'Price': item['price']
                                }
                                if 'quantity' in item and item['quantity'] != "1":
                                    row_data['Quantity'] = item['quantity']
                                items_df_data.append(row_data)
                            st.dataframe(items_df_data, use_container_width=True)
                        
                        st.markdown("---")
                        
                        # Totals
                        st.markdown("### 💰 Totals")
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
                    
                    else:
                        st.info("📊 Upload and analyze a receipt to see structured data extraction!")
                
                with tab4:
                    st.markdown("### 🔧 Template Management")
                    
                    # Show current templates
                    st.markdown("#### Current Templates:")
                    for template in template_manager.get_all_templates():
                        st.write(f"• **{template['name']}** (ID: {template['id']})")
                    
                    st.markdown("---")
                    st.markdown("#### Test Template Matching")
                    
                    if hasattr(st.session_state, 'extracted_text'):
                        matched_template = template_manager.find_matching_template(st.session_state.extracted_text)
                        if matched_template:
                            st.success(f"✅ **Matched Template:** {matched_template.vendor_name}")
                            st.write(f"**Vendor ID:** {matched_template.vendor_id}")
                            st.write(f"**Recognition Patterns:** {[p.pattern for p in matched_template.recognition_patterns]}")
                        else:
                            st.warning("❌ **No Template Match** - Receipt did not match any available templates")
                            st.info("💡 This receipt would benefit from a custom template for improved accuracy.")
                    
                    st.markdown("---")
                    st.info("🚀 **Template Features:**\n- Automatic vendor detection\n- Enhanced item parsing\n- Improved accuracy for known formats\n- Extensible template system")
                
                # Download options
                st.markdown("---")
                st.subheader("💾 Download Results")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    # Download as text file
                    st.download_button(
                        label="📄 Download as Text",
                        data=st.session_state.extracted_text,
                        file_name="receipt_text.txt",
                        mime="text/plain"
                    )
                
                with col2:
                    # Download as JSON
                    if st.session_state.analysis:
                        json_data = json.dumps(st.session_state.analysis, indent=2)
                        st.download_button(
                            label="📋 Download Analysis",
                            data=json_data,
                            file_name="receipt_analysis.json",
                            mime="application/json"
                        )
                
                with col3:
                    # Download structured data
                    if hasattr(st.session_state, 'structured_data') and st.session_state.structured_data:
                        structured_json = json.dumps(st.session_state.structured_data, indent=2)
                        st.download_button(
                            label="🗂️ Download Structured Data",
                            data=structured_json,
                            file_name="receipt_structured.json",
                            mime="application/json"
                        )
                
                # Download visualization image if available
                if hasattr(st.session_state, 'visualization_image') and st.session_state.visualization_image:
                    st.markdown("---")
                    
                    # Save the visualization image to a bytes buffer
                    buf = io.BytesIO()
                    st.session_state.visualization_image.save(buf, format="PNG")
                    byte_im = buf.getvalue()
                    
                    st.download_button(
                        label="📸 Download Visualization Image",
                        data=byte_im,
                        file_name="receipt_visualization.png",
                        mime="image/png"
                    )
        
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")
    
    else:
        # Show instructions when no file is uploaded
        st.info("👆 Please upload a receipt image to get started")
        
        # Add some helpful tips
        with st.expander("📝 Tips for best OCR results"):
            st.markdown("""
            **For optimal text extraction:**
            - Use clear, well-lit photos
            - Ensure the entire receipt is visible and flat
            - Avoid shadows, glare, and reflections
            - Take photos straight-on (not at an angle)
            - Use higher resolution images when possible
            - Ensure good contrast between text and background
            
            **Supported formats:** PNG, JPG, JPEG
            """)
        
        # Explain the features
        with st.expander("🔍 What does this app do?"):
            st.markdown("""
            **This Receipt Scanner provides advanced vendor-specific parsing:**
            
            **1. 📄 Raw Text Tab:**
            - Shows the complete text extracted from your receipt
            - Useful for verification and manual review
            
            **2. 📊 Analysis Tab:**
            - Basic pattern detection for amounts, dates, and store info
            - Provides metrics and quick insights
            
            **3. 🗂️ Structured Data Tab:**
            - **Intelligent vendor-specific parsing** using templates
            - **Store Information:** Name, address, phone number
            - **Transaction Details:** Date, time, receipt number
            - **Itemized List:** Individual products and prices in a table
            - **Financial Summary:** Subtotal, tax, and total amounts
            - **Payment Method:** How the transaction was paid
            - **Enhanced Accuracy:** Better results for supported vendors
            
            **4. 🔧 Template Management:**
            - View available vendor templates
            - See which template matched your receipt
            - Test template matching functionality
            
            **5. 🔍 OCR Visualization:**
            - Shows all detected text with bounding boxes
            - Displays text labels for each detected area
            - Helps verify the accuracy of text recognition
            
            **🎯 Vendor Templates Supported:**
            - Walmart, Target, Costco, Kroger (and affiliated stores)
            - Enhanced parsing accuracy for these vendors
            - Automatic vendor detection and specialized field extraction
            """)

if __name__ == "__main__":
    main() 