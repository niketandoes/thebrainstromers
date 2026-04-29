import streamlit as st
import torch
from torchvision import transforms
from PIL import Image

# Streamlit Page Config
st.set_page_config(
    page_title="EuroSAT Land Cover Classifier",
    page_icon="🌍",
    layout="centered"
)

# Load saved model from disk — NO API calls
@st.cache_resource
def load_model():
    try:
        model = torch.load("model.pth", map_location="cpu")
        model.eval()
        return model
    except Exception as e:
        return None

model = load_model()

# Header
st.title("🌍 EuroSAT Land Cover Classifier")
st.markdown("### KaggleHacX '26 — Team Submission")
st.write("Upload a satellite image to classify the land cover type. This model is trained on the **EuroSAT dataset** and runs entirely locally without any external API calls.")

st.divider()

# Layout
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Upload Image")
    uploaded = st.file_uploader("Choose a satellite image...", type=["jpg", "jpeg", "png"])

with col2:
    st.subheader("Prediction Results")
    if uploaded:
        if model:
            image = Image.open(uploaded).convert("RGB")
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Preprocessing
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
            tensor = transform(image).unsqueeze(0)
            
            # Inference
            with torch.no_grad():
                output = model(tensor)
                prob = torch.softmax(output, dim=1)
                pred = prob.argmax().item()
                confidence = prob.max().item()
            
            # EuroSAT Classes
            CLASS_NAMES = [
                "Annual Crop", "Forest", "Herbaceous Vegetation", 
                "Highway", "Industrial", "Pasture", 
                "Permanent Crop", "Residential", "River", "Sea/Lake"
            ]
            
            st.success(f"**Prediction:** {CLASS_NAMES[pred]}")
            st.info(f"**Confidence:** {confidence:.2%}")
            
            # Show a progress bar for confidence
            st.progress(confidence)
        else:
            st.error("⚠️ Model 'model.pth' not found or could not be loaded. Please run the training script first.")
    else:
        st.info("Please upload an image to see the prediction.")
