import os 
os.add_dll_directory(r"C:\ProgramData\anaconda3\Lib\site-packages\openslide-bin-4.0.0.6-windows-x64\bin")
import os 
import streamlit as st
import torch
from PIL import Image
import numpy as np
from openslide import open_slide
from openslide.deepzoom import DeepZoomGenerator
import shutil
import matplotlib.pyplot as plt
import glob

# Import the new predictor
from hcnn_predictor import HCNNPredictor

# Custom page configuration
st.set_page_config(
    page_title="Histopathology Analysis",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .stApp {
        max-width: 1200px;
        margin: 0 auto;
    }
    .disease-prediction {
        padding: 20px;
        border-radius: 5px;
        background-color: #f0f2f6;
        margin: 10px 0;
        text-align: center;
    }
    .disease-name {
        font-size: 24px;
        font-weight: bold;
        color: #2c3e50;
        margin-bottom: 10px;
    }
    .confidence-score {
        font-size: 18px;
        color: #34495e;
    }
    .disease-description {
        background-color: #f8f9fa;
        border-left: 4px solid #007bff;
        padding: 15px;
        margin-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Disease Descriptions Dictionary
DISEASE_DESCRIPTIONS = {
    "Odontogenic Keratocyst": {
        "description": "An odontogenic keratocyst (OKC) is a rare, benign but locally aggressive cystic lesion that originates from dental lamina remnants. It is characterized by its thin, fragile wall and high recurrence rate. This type of cyst typically develops in the jaw bone and can grow slowly over time. The cyst is lined with keratinized epithelium and has a tendency to recur after surgical removal.",
        "key_features": [
            "Most commonly found in the lower jaw (mandible)",
            "Typically affects individuals between 20-40 years old",
            "Can be associated with genetic conditions like Gorlin syndrome"
        ],
        "clinical_significance": "Requires careful surgical management due to high recurrence risk and potential for local invasion. Regular follow-up and monitoring are crucial to prevent repeated development."
    },
    "Dentigerous Cyst": {
        "description": "A dentigerous cyst is a developmental odontogenic cyst that surrounds the crown of an unerupted tooth. It forms when fluid accumulates between the reduced enamel epithelium and the crown of an unerupted tooth. These cysts are typically associated with impacted or unerupted teeth and can cause displacement of adjacent teeth if left untreated.",
        "key_features": [
            "Most common type of developmental odontogenic cyst",
            "Usually affects impacted or unerupted teeth",
            "Typically discovered during routine dental radiographic examinations"
        ],
        "clinical_significance": "Generally benign and can be treated by surgical removal of the cyst and associated tooth. Early detection prevents potential complications like tooth displacement or bone resorption."
    },
    "Dental Follicle": {
        "description": "A dental follicle is a soft tissue surrounding the developing tooth crown before tooth eruption. It plays a crucial role in tooth development and contains the cells that will form important dental structures. The follicle is a transitional tissue that is essential for the normal formation and eruption of teeth, and can sometimes develop into pathological conditions if abnormal changes occur.",
        "key_features": [
            "Contains cells that will form periodontal ligament, cementum, and alveolar bone",
            "Critical in the process of tooth formation and eruption",
            "Can develop into various pathological conditions if abnormal"
        ],
        "clinical_significance": "Important in understanding tooth development and potential developmental anomalies. Abnormalities in the dental follicle can lead to issues with tooth formation, eruption, or potential cystic developments."
    }
}

def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        "temp_upload",
        "output_tiles",
        "classified_tiles/empty",
        "classified_tiles/connective",
        "classified_tiles/okc_epithelial",
        "classified_tiles/dc_epithelial",
        "classified_tiles/df_epithelial"
    ]
    for dir_path in directories:
        os.makedirs(dir_path, exist_ok=True)

def get_thumbnail(slide, max_size=(1000, 1000)):
    """
    Generate a thumbnail of the slide using a lower resolution level.
    Args:
        slide: OpenSlide object.
        max_size: Maximum size of the thumbnail (width, height).
    Returns:
        Thumbnail image as a PIL.Image object.
    """
    # Find the best level for downsampling to reduce memory usage
    best_level = slide.get_best_level_for_downsample(max(
        slide.dimensions[0] / max_size[0], 
        slide.dimensions[1] / max_size[1]
    ))
    
    # Get dimensions of the best level
    level_dims = slide.level_dimensions[best_level]
    
    try:
        # Read region at the best level with reduced memory footprint
        thumbnail = slide.read_region((0, 0), best_level, level_dims)
        return thumbnail.convert("RGB")
    
    except MemoryError:
        # Fallback strategy: Use Pillow to resize if OpenSlide fails
        st.warning("Large image detected. Using alternative thumbnail generation method.")
        base_image = slide.read_region((0, 0), slide.level_count - 1, slide.level_dimensions[-1])
        base_image = base_image.convert("RGB")
        base_image.thumbnail(max_size, Image.LANCZOS)
        return base_image

def main():
    st.title("🔬 Histopathology Analysis System")
    
    create_directories()
    
    # Initialize the predictor
    try:
        predictor = HCNNPredictor(model_path="best_enhanced_hcnn_model.pth")
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return

    # Custom file uploader to handle large files
    st.write("Upload TIFF Image (max 4GB)")
    uploaded_file = st.file_uploader("", type=["tiff", "tif"], accept_multiple_files=False)

    if uploaded_file is not None:
        file_size = uploaded_file.size
        max_size = 4 * 1024 * 1024 * 1024  # 4 GB
        
        if file_size > max_size:
            st.error(f"File is too large. Maximum file size is 4 GB. Your file is {file_size / (1024 * 1024 * 1024):.2f} GB.")
        else:
            if uploaded_file.name not in st.session_state.processed_files:
                st.session_state.processed_files.append(uploaded_file.name)
                
                # Save uploaded file
                temp_path = os.path.join("temp_upload", uploaded_file.name)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Display thumbnail of slide
                st.subheader("Slide Thumbnail")
                slide = open_slide(temp_path)
                thumbnail = get_thumbnail(slide)
                st.image(thumbnail, caption="Tissue Slide Thumbnail", use_container_width=True)
                
                with st.spinner("Analyzing tissue slide..."):
                    prediction_result = predictor.predict(temp_path)
                
                st.subheader("Tissue Classification Results")
                
                if prediction_result.get('prediction_successful'):
                    disease = prediction_result.get('disease', 'Unknown')
                    confidence = prediction_result.get('confidence', 0)
                    
                    st.markdown(f"""
                        <div class="disease-prediction">
                            <div class="disease-name">{disease}</div>
                            <div class="confidence-score">Confidence: {confidence:.2f}%</div>
                        </div>
                    """, unsafe_allow_html=True)

                else:
                    st.error(f"Prediction failed: {prediction_result.get('error', 'Unknown error')}")

                # Optional: Add visualization or additional information
                st.subheader("Additional Information")
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Image Type", "Whole Slide Image" if temp_path.lower().endswith(('.tiff', '.tif')) else "Single Image")
                with col2:
                    st.metric("File Size", f"{os.path.getsize(temp_path) / (1024 * 1024):.2f} MB")
                
                # Add Disease Description if available
                if prediction_result.get('prediction_successful') and disease in DISEASE_DESCRIPTIONS:
                    disease_info = DISEASE_DESCRIPTIONS[disease]
                    
                    # Display disease description with styled div
                    st.markdown(f"""
                        <div class="disease-description">
                            <h4>Disease Description: {disease}</h4>
                            <p>{disease_info['description']}</p>
                            <h5>Key Features:</h5>
                            <ul>
                                {''.join(f'<li>{feature}</li>' for feature in disease_info['key_features'])}
                            </ul>
                            <h5>Clinical Significance:</h5>
                            <p>{disease_info['clinical_significance']}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                try:
                    os.remove(temp_path)
                except Exception as e:
                    st.warning(f"Could not remove temporary file: {e}")

def display_help():
    st.sidebar.header("🆘 Help & Instructions")
    st.sidebar.markdown("""
    ### How to Use the Histopathology Analysis System
    
    1. **Upload Image**: Use the file uploader to select a TIFF image.
    2. **Image Processing**: The system generates a thumbnail and analyzes tissue.
    3. **Results Interpretation**: Displays predicted disease type and confidence.
    
    ### Supported Tissue Types
    - Odontogenic Keratocyst
    - Dentigerous Cyst
    - Dental Follicle
    """)

if __name__ == "__main__":
    display_help()
    main()