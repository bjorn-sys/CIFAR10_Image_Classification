import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import io
import os
import base64

# ==============================================================
# MODEL DEFINITION: AlexNetMiniLite (Same as training)
# ==============================================================
class AlexNetMiniLite(nn.Module):
    def __init__(self, num_classes=10):
        super(AlexNetMiniLite, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=0.4),
            nn.Linear(in_features=128 * 4 * 4, out_features=256),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=256, out_features=num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ==============================================================
# MODEL LOADING FUNCTION
# ==============================================================
@st.cache_resource
def load_model():
    model = AlexNetMiniLite(num_classes=10)
    try:
        # Load the pre-trained weights
        model.load_state_dict(torch.load('Alexnet_model.pth', map_location=torch.device('cpu')))
    except:
        st.error("Model file 'Alexnet_model.pth' not found. Please ensure the model file is in the same directory.")
        return None
    model.eval()
    return model

# ==============================================================
# IMAGE PREPROCESSING
# ==============================================================
def preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])
    return transform(image).unsqueeze(0)

# ==============================================================
# PREDICTION FUNCTION
# ==============================================================
def predict_image(model, image_tensor):
    with torch.no_grad():
        outputs = model(image_tensor)
        _, predicted = torch.max(outputs, 1)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
    return predicted.item(), probabilities[0].numpy()

# ==============================================================
# CIFAR-10 CLASS NAMES
# ==============================================================
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck']

# ==============================================================
# AUTHENTICATION
# ==============================================================
def check_login(username, password):
    # Get credentials from environment variables or use defaults
    admin_user = os.getenv('APP_USERNAME', 'admin')
    admin_password = os.getenv('APP_PASSWORD', 'admin123')
    
    # Simple authentication
    valid_users = {
        admin_user: admin_password
    }
    return username in valid_users and valid_users[username] == password

# ==============================================================
# CUSTOM CAMERA COMPONENT WITH BACK CAMERA SUPPORT
# ==============================================================
def camera_component_with_back_camera():
    """Custom camera component that allows back camera selection"""
    
    st.markdown("""
    <style>
    .camera-options {
        margin-bottom: 10px;
    }
    .camera-instructions {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
        margin: 10px 0;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Camera selection
    col1, col2 = st.columns([1, 2])
    with col1:
        camera_mode = st.radio(
            "Camera Mode",
            ["Front Camera", "Back Camera"],
            help="Select which camera to use"
        )
    
    with col2:
        st.markdown(f"""
        <div class="camera-instructions">
        📱 <strong>Using {camera_mode}:</strong><br>
        • For better results, ensure good lighting<br>
        • Hold steady while capturing<br>
        • Position object in the center
        </div>
        """, unsafe_allow_html=True)
    
    # JavaScript to switch cameras
    camera_constraints = {
        "Front Camera": {"facingMode": "user"},
        "Back Camera": {"facingMode": "environment"}
    }
    
    # HTML and JavaScript for custom camera component
    camera_html = f"""
    <script>
    let currentStream = null;
    
    async function setupCamera(facingMode) {{
        if (currentStream) {{
            currentStream.getTracks().forEach(track => track.stop());
        }}
        
        const constraints = {{
            video: {{ 
                facingMode: facingMode,
                width: {{ ideal: 1280 }},
                height: {{ ideal: 720 }}
            }} 
        }};
        
        try {{
            currentStream = await navigator.mediaDevices.getUserMedia(constraints);
            const video = document.getElementById('video');
            video.srcObject = currentStream;
        }} catch (err) {{
            console.error("Error accessing camera:", err);
            alert("Error accessing camera: " + err.message);
        }}
    }}
    
    function capturePhoto() {{
        const video = document.getElementById('video');
        const canvas = document.getElementById('canvas');
        const context = canvas.getContext('2d');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        const dataURL = canvas.toDataURL('image/jpeg');
        document.getElementById('photoData').value = dataURL;
        document.getElementById('captureForm').submit();
    }}
    
    // Initialize with front camera
    setupCamera('{camera_constraints[camera_mode]["facingMode"]}');
    
    // Update camera when selection changes
    document.getElementById('cameraMode').addEventListener('change', function(e) {{
        const facingMode = e.target.value === 'Back Camera' ? 'environment' : 'user';
        setupCamera(facingMode);
    }});
    </script>
    
    <div style="text-align: center;">
        <video id="video" autoplay playsinline style="width: 100%; max-width: 500px; border-radius: 10px; background: #000;"></video>
        <canvas id="canvas" style="display: none;"></canvas>
        
        <div style="margin: 15px 0;">
            <button onclick="capturePhoto()" style="
                background: #ff4b4b;
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 25px;
                font-size: 16px;
                cursor: pointer;
                margin: 10px;
            ">📸 Capture Image</button>
        </div>
    </div>
    
    <form id="captureForm">
        <input type="hidden" id="photoData" name="photoData">
        <input type="hidden" id="cameraMode" value="{camera_mode}">
    </form>
    """
    
    st.components.v1.html(camera_html, height=600)
    
    # Handle captured image
    if st.session_state.get('photo_data'):
        try:
            # Decode base64 image
            image_data = st.session_state.photo_data.split(',')[1]
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
            return image
        except Exception as e:
            st.error(f"Error processing captured image: {e}")
            return None
    
    return None

# ==============================================================
# STREAMLIT APP
# ==============================================================
def main():
    st.set_page_config(
        page_title="CIFAR-10 Image Classifier",
        page_icon="🖼️",
        layout="wide"
    )
    
    # Initialize session state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    if 'username' not in st.session_state:
        st.session_state.username = ''
    if 'photo_data' not in st.session_state:
        st.session_state.photo_data = None
    
    # Handle form submission for captured images
    if st.query_params.get('photoData'):
        st.session_state.photo_data = st.query_params.get('photoData')
        st.rerun()
    
    # Login section
    if not st.session_state.authenticated:
        st.title("🔐 CIFAR-10 Image Classifier Login")
        
        with st.form("login_form"):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_button = st.form_submit_button("Login")
            
            if login_button:
                if username and password:
                    if check_login(username, password):
                        st.session_state.authenticated = True
                        st.session_state.username = username
                        st.success(f"Welcome {username}!")
                        st.rerun()
                    else:
                        st.error("Invalid username or password")
                else:
                    st.error("Please enter both username and password")
        
        # Contact information instead of demo credentials
        st.markdown("---")
        st.markdown("**Need access?**")
        st.markdown("Please contact the system administrator for login credentials.")
        
        return
    
    # Main application after login
    st.sidebar.title("Navigation")
    app_mode = st.sidebar.selectbox("Choose Input Method", 
                                   ["Upload Image", "Camera Input"])
    
    st.sidebar.markdown("---")
    st.sidebar.write(f"Logged in as: **{st.session_state.username}**")
    if st.sidebar.button("Logout"):
        st.session_state.authenticated = False
        st.session_state.username = ''
        st.session_state.photo_data = None
        st.rerun()
    
    st.title("🖼️ CIFAR-10 Image Classification")
    st.write(f"Welcome, **{st.session_state.username}**!")
    
    # Load model
    model = load_model()
    if model is None:
        return
    
    # Input method selection
    if app_mode == "Upload Image":
        st.header("Upload an Image")
        uploaded_file = st.file_uploader(
            "Choose an image file", 
            type=['jpg', 'jpeg', 'png', 'bmp'],
            help="Upload an image to classify (will be resized to 32x32)"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file).convert('RGB')
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocess and predict
            image_tensor = preprocess_image(image)
            prediction, probabilities = predict_image(model, image_tensor)
            
            # Display results
            st.subheader("🔍 Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", class_names[prediction])
                st.write(f"Class Index: {prediction}")
            
            with col2:
                confidence = probabilities[prediction] * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show probability distribution
            st.subheader("📊 Class Probabilities")
            prob_data = {class_names[i]: float(probabilities[i]) * 100 for i in range(10)}
            sorted_probs = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))
            
            for class_name, prob in sorted_probs.items():
                st.write(f"**{class_name}**: {prob:.2f}%")
                st.progress(float(prob) / 100)
    
    else:  # Camera Input
        st.header("📱 Camera Input")
        
        # Clear previous captured image if switching modes
        if st.session_state.photo_data:
            st.session_state.photo_data = None
        
        # Use custom camera component
        captured_image = camera_component_with_back_camera()
        
        if captured_image is not None:
            # Display captured image
            st.image(captured_image, caption="Captured Image", use_column_width=True)
            
            # Preprocess and predict
            image_tensor = preprocess_image(captured_image)
            prediction, probabilities = predict_image(model, image_tensor)
            
            # Display results
            st.subheader("🔍 Prediction Results")
            col1, col2 = st.columns(2)
            
            with col1:
                st.metric("Predicted Class", class_names[prediction])
                st.write(f"Class Index: {prediction}")
            
            with col2:
                confidence = probabilities[prediction] * 100
                st.metric("Confidence", f"{confidence:.2f}%")
            
            # Show probability distribution
            st.subheader("📊 Class Probabilities")
            prob_data = {class_names[i]: float(probabilities[i]) * 100 for i in range(10)}
            sorted_probs = dict(sorted(prob_data.items(), key=lambda x: x[1], reverse=True))
            
            for class_name, prob in sorted_probs.items():
                st.write(f"**{class_name}**: {prob:.2f}%")
                st.progress(float(prob) / 100)
            
            # Option to capture another image
            if st.button("🔄 Capture Another Image"):
                st.session_state.photo_data = None
                st.rerun()
    
    # Information section
    st.markdown("---")
    st.subheader("ℹ️ About CIFAR-10")
    st.write("""
    The CIFAR-10 dataset consists of 60,000 32x32 color images in 10 classes, with 6,000 images per class. 
    The classes are: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, and truck.
    
    **Note:** The model expects 32x32 pixel images. Uploaded/captured images will be automatically resized.
    """)

if __name__ == "__main__":
    main()