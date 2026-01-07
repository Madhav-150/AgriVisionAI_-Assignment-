import streamlit as st
import torch
from PIL import Image
from src import config, dataset, model
import time

# Page Config
st.set_page_config(
    page_title="AgriVision AI",
    page_icon="🌿",
    layout="centered"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main {
        background-color: #f5f7f9;
        font-family: 'Inter', sans-serif;
    }
    .stButton>button {
        width: 100%;
        background-color: #2e7d32;
        color: white;
        border-radius: 8px;
        height: 50px;
        font-weight: 600;
        border: none;
    }
    .stButton>button:hover {
        background-color: #1b5e20;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        text-align: center;
        margin-top: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .healthy {
        background-color: #e8f5e9;
        color: #1b5e20;
        border: 2px solid #2e7d32;
    }
    .diseased {
        background-color: #ffebee;
        color: #c62828;
        border: 2px solid #d32f2f;
    }
    h1 {
        color: #1b5e20;
        text-align: center;
        font-weight: 800;
    }
    </style>
""", unsafe_allow_html=True)

# Main Header
st.title("🌿 AgriVision AI")
st.markdown("<p style='text-align: center; color: #555;'>Smart Leaf Disease Detection System</p>", unsafe_allow_html=True)
st.divider()

# Load Model (Cached)
@st.cache_resource
def load_model():
    # Attempt to load trained model
    try:
        classes = config.CLASSES
        net = model.build_model(num_classes=len(classes))
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
        net = net.to(config.DEVICE)
        net.eval()
        return net, classes
    except FileNotFoundError:
        return None, None

net, classes = load_model()

# Create columns for layout
col1, col2 = st.columns([1, 2])

with col1:
    st.image("https://cdn-icons-png.flaticon.com/512/628/628283.png", width=100) # Leaf icon placeholder
    st.info("Upload an image of a crop leaf to detect if it is healthy or diseased.")

with col2:
    uploaded_file = st.file_uploader("Choose a leaf image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    st.divider()
    
    # Display Image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', width="stretch")
    
    if st.button("Analyze Leaf Health"):
        if net is None:
            st.error("Model file not found! Please train the model first by running `python model_train.py`.")
        else:
            with st.spinner('Analyzing patterns...'):
                time.sleep(1) # Simulating processing time for effect
                
                # Preprocess
                transforms = dataset.get_transforms('test')
                img_t = transforms(image).unsqueeze(0).to(config.DEVICE)
                
                # Predict
                with torch.no_grad():
                    outputs = net(img_t)
                    probs = torch.nn.functional.softmax(outputs, dim=1)
                    conf, predicted = torch.max(probs, 1)
                
                class_name = classes[predicted.item()]
                confidence = conf.item() * 100
                
                # Result Display
                css_class = "healthy" if class_name.lower() == "healthy" else "diseased"
                
                st.markdown(f"""
                    <div class="prediction-box {css_class}">
                        <h2>Prediction: {class_name.upper()}</h2>
                        <p>Confidence Score: <b>{confidence:.2f}%</b></p>
                    </div>
                """, unsafe_allow_html=True)
                
                if class_name.lower() == "diseased":
                    st.warning("⚠️ Action Recommended: Inspect crop for potential spread.")
                else:
                    st.success("✅ Specimen appears healthy.")

# Footer
st.markdown("---")
st.markdown("<p style='text-align: center; font-size: 12px; color: #888;'>Powered by MobileNetV2 & PyTorch for AgriVision AI</p>", unsafe_allow_html=True)
