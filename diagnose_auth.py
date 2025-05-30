#!/usr/bin/env python3
"""
Google Cloud Services Authentication Diagnostic Tool
This script helps diagnose authentication issues with Google Cloud Vision API and Vertex AI.
"""

import os
import json
import sys
import base64
from google.cloud import vision
from google.oauth2 import service_account
from google.auth.exceptions import GoogleAuthError
import vertexai
from vertexai.generative_models import GenerativeModel, Part

def load_and_validate_credentials():
    """Load and validate the service account credentials."""
    credentials_path = "pg-cashback-5290642eb30b.json"
    
    print("🔍 Step 1: Checking credentials file...")
    
    # Check if file exists
    if not os.path.exists(credentials_path):
        print(f"❌ ERROR: Credentials file not found: {credentials_path}")
        return None, None
    
    print(f"✅ Credentials file found: {credentials_path}")
    
    # Try to load and parse JSON
    try:
        with open(credentials_path, 'r') as f:
            cred_data = json.load(f)
        print("✅ Credentials file is valid JSON")
    except json.JSONDecodeError as e:
        print(f"❌ ERROR: Invalid JSON in credentials file: {e}")
        return None, None
    
    # Validate required fields
    required_fields = [
        'type', 'project_id', 'private_key_id', 'private_key',
        'client_email', 'client_id', 'auth_uri', 'token_uri'
    ]
    
    missing_fields = [field for field in required_fields if field not in cred_data]
    if missing_fields:
        print(f"❌ ERROR: Missing required fields: {missing_fields}")
        return None, None
    
    print("✅ All required fields present in credentials")
    print(f"   - Project ID: {cred_data['project_id']}")
    print(f"   - Service Account Email: {cred_data['client_email']}")
    
    return cred_data, credentials_path

def test_credentials_loading():
    """Test if credentials can be loaded by Google Auth library."""
    print("\n🔍 Step 2: Testing credentials loading...")
    
    cred_data, credentials_path = load_and_validate_credentials()
    if not cred_data:
        return None, None
    
    try:
        credentials = service_account.Credentials.from_service_account_file(
            credentials_path,
            scopes=['https://www.googleapis.com/auth/cloud-platform']
        )
        print("✅ Credentials loaded successfully")
        return credentials, cred_data
    except Exception as e:
        print(f"❌ ERROR: Failed to load credentials: {e}")
        return None, None

def test_vision_client_creation(credentials):
    """Test if Vision client can be created."""
    print("\n🔍 Step 3: Testing Vision client creation...")
    
    try:
        client = vision.ImageAnnotatorClient(credentials=credentials)
        print("✅ Vision client created successfully")
        return client
    except Exception as e:
        print(f"❌ ERROR: Failed to create Vision client: {e}")
        return None

def test_vertex_ai_setup(credentials, cred_data):
    """Test if Vertex AI can be initialized."""
    print("\n🔍 Step 4: Testing Vertex AI setup...")
    
    try:
        project_id = cred_data['project_id']
        
        # Initialize Vertex AI
        vertexai.init(
            project=project_id,
            location="us-central1",
            credentials=credentials
        )
        
        # Initialize the Gemini model
        model = GenerativeModel('gemini-2.0-flash-exp')
        print("✅ Vertex AI initialized successfully")
        return model
    except Exception as e:
        print(f"❌ ERROR: Failed to initialize Vertex AI: {e}")
        return None

def test_vision_api_call(client):
    """Test a simple Vision API call to check authentication."""
    print("\n🔍 Step 5: Testing Vision API call...")
    
    try:
        # Create a simple test image (1x1 pixel)
        test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x07Z\xbc\x00\x00\x00\x00IEND\xaeB`\x82'
        
        image = vision.Image(content=test_image_data)
        response = client.text_detection(image=image)
        
        if response.error.message:
            print(f"❌ Vision API Error: {response.error.message}")
            return False
        else:
            print("✅ Vision API call successful")
            return True
            
    except Exception as e:
        print(f"❌ ERROR: Vision API call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Check for specific error types
        if "401" in str(e):
            print("   💡 This is a 401 authentication error")
            print("   💡 Possible causes:")
            print("      - Vision API not enabled for your project")
            print("      - Service account lacks necessary permissions")
            print("      - Invalid or expired credentials")
        
        return False

def test_vertex_ai_call(model):
    """Test a simple Vertex AI call to check authentication."""
    print("\n🔍 Step 6: Testing Vertex AI call...")
    
    try:
        # Simple text-only test first to avoid image complications
        text_response = model.generate_content("Hello! Just say 'API working' if you can see this.")
        
        if text_response.text:
            print("✅ Vertex AI text generation successful")
            print(f"   Response: {text_response.text}")
            
            # Now try with a simple test image
            print("   Testing image processing capability...")
            
            # Create a simple 1x1 pixel image
            test_image_data = b'\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00\x00\x00\tpHYs\x00\x00\x0b\x13\x00\x00\x0b\x13\x01\x00\x9a\x9c\x18\x00\x00\x00\nIDATx\x9cc\xf8\x00\x00\x00\x01\x00\x01\x00\x00\x00\x00\x00\x00\x07Z\xbc\x00\x00\x00\x00IEND\xaeB`\x82'
            
            # Create a Part from the image data
            b64_image = base64.b64encode(test_image_data).decode('utf-8')
            image_part = Part.from_data(
                data=base64.b64decode(b64_image),
                mime_type="image/png"
            )
            
            # Simple prompt for image test
            try:
                image_response = model.generate_content(
                    ["Describe this test image briefly.", image_part]
                )
                print("✅ Vertex AI image processing successful")
                return True
            except Exception as img_err:
                print(f"⚠️ Vertex AI image processing test failed: {img_err}")
                print("   Note: Text generation still works, so basic API is functional")
                return True  # Still return True as text generation worked
            
        else:
            print("❌ Vertex AI call failed: No response text")
            return False
            
    except Exception as e:
        print(f"❌ ERROR: Vertex AI call failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        
        # Check for specific error types
        if "401" in str(e) or "403" in str(e):
            print("   💡 This is an authentication/authorization error")
            print("   💡 Possible causes:")
            print("      - Vertex AI API not enabled for your project")
            print("      - Service account lacks Vertex AI permissions")
            print("      - Invalid or expired credentials")
        
        return False

def provide_setup_guidance():
    """Provide guidance on setting up the APIs."""
    print("\n🔧 Setup Guidance:")
    
    print("\n1. Enable required APIs:")
    print("   gcloud services enable vision.googleapis.com --project=pg-cashback")
    print("   gcloud services enable aiplatform.googleapis.com --project=pg-cashback")
    
    print("\n2. Grant necessary permissions to service account:")
    print("   gcloud projects add-iam-policy-binding pg-cashback \\")
    print("     --member='serviceAccount:pg-cashback@pg-cashback.iam.gserviceaccount.com' \\")
    print("     --role='roles/ml.admin'")
    
    print("   gcloud projects add-iam-policy-binding pg-cashback \\")
    print("     --member='serviceAccount:pg-cashback@pg-cashback.iam.gserviceaccount.com' \\")
    print("     --role='roles/aiplatform.user'")
    
    print("\n3. Alternative: More specific roles:")
    print("   - Cloud Vision API User: roles/ml.developer")
    print("   - Vertex AI User: roles/aiplatform.user")
    
    print("\n4. If issues persist:")
    print("   - Check that APIs are enabled in Google Cloud Console")
    print("   - Verify the service account has the necessary permissions")
    print("   - Try regenerating the service account key")

def main():
    """Run the complete diagnostic."""
    print("🔧 Google Cloud Services Authentication Diagnostic")
    print("=" * 60)
    print("Testing both Vision API and Vertex AI with the same credentials")
    
    # Test credentials loading
    credentials, cred_data = test_credentials_loading()
    if not credentials:
        print("\n❌ Cannot proceed without valid credentials")
        provide_setup_guidance()
        return
    
    # Test Vision client creation
    vision_client = test_vision_client_creation(credentials)
    
    # Test Vertex AI setup
    vertex_model = test_vertex_ai_setup(credentials, cred_data)
    
    # Test API calls
    vision_success = False
    vertex_success = False
    
    if vision_client:
        vision_success = test_vision_api_call(vision_client)
    
    if vertex_model:
        vertex_success = test_vertex_ai_call(vertex_model)
    
    # Summary
    print("\n📊 Summary:")
    print(f"   Vision API: {'✅ Working' if vision_success else '❌ Failed'}")
    print(f"   Vertex AI:  {'✅ Working' if vertex_success else '❌ Failed'}")
    
    if not vision_success or not vertex_success:
        provide_setup_guidance()
    else:
        print("\n🎉 All tests passed! Both Vision API and Vertex AI are working.")
        print("   You can now run the receipt analysis app:")
        print("   streamlit run app.py")

if __name__ == "__main__":
    main() 