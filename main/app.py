import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
import pandas as pd
import numpy as np

# Streamlit Page Config
st.set_page_config(
    page_title="EuroSAT Land Cover Classifier",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for the advanced UI
st.markdown("""
    <style>
    /* Hide Streamlit Default Elements */
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Global Theme - Dark gradient background */
    .stApp, [data-testid="stAppViewContainer"] {
        background: radial-gradient(circle at 50% 110%, rgba(0, 102, 255, 0.8) 0%, rgba(5, 5, 20, 1) 50%, rgba(0, 0, 0, 1) 100%) !important;
        background-attachment: fixed !important;
        color: #ffffff;
        font-family: 'Inter', sans-serif;
    }
    
    /* Ensure Streamlit's inner block doesn't override the background */
    .main .block-container {
        background: transparent !important;
    }
    
    /* Navbar styling */
    .custom-navbar {
        display: flex;
        align-items: center;
        padding: 20px 50px;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        margin-top: -4rem;
        margin-bottom: 4rem;
    }
    .nav-left, .nav-right {
        display: flex;
        gap: 20px;
        font-size: 14px;
        color: #a0a0a0;
        flex: 1;
        align-items: center;
    }
    .nav-right {
        justify-content: flex-end;
    }
    .nav-logo {
        font-size: 20px;
        font-weight: bold;
        color: white;
        text-align: center;
    }
    .nav-btn {
        background-color: white;
        color: black;
        padding: 8px 20px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 14px;
    }
    
    /* Hero Section Styling */
    .hero-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        text-align: center;
        margin-bottom: 60px;
    }
    .pill-badge {
        background-color: rgba(0, 100, 255, 0.2);
        color: #4da6ff;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-bottom: 20px;
        border: 1px solid rgba(0, 100, 255, 0.5);
    }
    .hero-title {
        font-size: 5rem;
        font-weight: 800;
        line-height: 1.1;
        margin: 0 0 20px 0;
        letter-spacing: -2px;
    }
    .hero-subtitle {
        color: #a0a0a0;
        max-width: 600px;
        font-size: 16px;
        margin-bottom: 30px;
    }
    
    /* Card / Container Styling for the app */
    div[data-testid="stVerticalBlock"] > div[style*="flex-direction: column;"] {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 16px;
        padding: 20px;
        backdrop-filter: blur(10px);
    }
    
    /* Streamlit overrides */
    h2, h3, p {
        color: white !important;
    }
    .stFileUploader > div > div {
        background: rgba(0, 0, 0, 0.3) !important;
        border: 1px dashed rgba(255, 255, 255, 0.2) !important;
    }
    div[data-testid="stMetricValue"] {
        font-size: 2.5rem;
        color: #4da6ff;
    }
    </style>
""", unsafe_allow_html=True)

# Custom HTML Header / Navbar / Hero
st.markdown("""
<div class="custom-navbar">
    <div class="nav-left">
        <span>About</span>
        <span>Technologies</span>
        <span>Models</span>
    </div>
    <div class="nav-logo">EuroSAT.AI</div>
    <div class="nav-right">
        <span>Team</span>
        <span class="nav-btn">Get Started</span>
    </div>
</div>

<div class="hero-container">
    <div class="pill-badge">AI Powered Earth Observation</div>
    <h1 class="hero-title">Smart.<br>Classified.<br>Future-Ready.</h1>
    <p class="hero-subtitle">EuroSAT.AI is your one-stop platform for satellite imagery analysis. It is the optimal deep learning solution to get a clear view of our changing world.</p>
</div>
""", unsafe_allow_html=True)

# Load saved model from disk
@st.cache_resource(show_spinner=False)
def load_model():
    try:
        model = torch.load("model.pth", map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model()

# --- EuroSAT Classes ---
CLASS_NAMES = [
    "Annual Crop", "Forest", "Herbaceous Vegetation", 
    "Highway", "Industrial", "Pasture", 
    "Permanent Crop", "Residential", "River", "Sea/Lake"
]

# --- Main Layout ---
# We use a container to wrap the interactive part
st.markdown("<h3 style='text-align: center;'>Try the Demo</h3><br>", unsafe_allow_html=True)

col1, col2 = st.columns([1, 1.5], gap="large")

with col1:
    st.markdown("### 1. Upload Image")
    st.markdown("<p style='color: #a0a0a0; font-size: 14px;'>Choose a .jpg, .jpeg, or .png Sentinel-2 image.</p>", unsafe_allow_html=True)
    uploaded = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
    
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)

with col2:
    st.markdown("### 2. Analysis Results")
    
    if not uploaded:
        st.info("Please upload a satellite image to see the model's prediction.")
    elif not model:
        st.error("⚠️ Model 'model.pth' not found or could not be loaded.")
    else:
        with st.spinner("Analyzing image..."):
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            tensor = transform(image).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1).squeeze().numpy()
            
            # Get top predictions
            top5_idx = np.argsort(prob)[-5:][::-1]
            top5_probs = prob[top5_idx]
            top5_classes = [CLASS_NAMES[i] for i in top5_idx]
            
            pred_class = top5_classes[0]
            pred_conf = top5_probs[0]
            
            st.success("Analysis Complete!")
            metric_col1, metric_col2 = st.columns(2)
            metric_col1.metric("Predicted Class", pred_class)
            metric_col2.metric("Confidence", f"{pred_conf:.2%}")
            
            st.divider()
            
            st.markdown("<p style='color: #a0a0a0;'>Probability Distribution (Top 5)</p>", unsafe_allow_html=True)
            
            df = pd.DataFrame({
                "Class": top5_classes,
                "Probability": top5_probs
            })
            
            st.bar_chart(df.set_index("Class"), color="#4da6ff")

