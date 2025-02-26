import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import yaml
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple, Optional
import warnings

# Disable all warnings
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def load_class_names(dataset_yaml: str = "Data/3_WebUI_7k/yolo_dataset/dataset.yaml") -> Dict[int, str]:
    """Load class names from dataset yaml file."""
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def visualize_predictions(
    image,
    results,
    class_names: Dict[int, str],
    conf_threshold: float = 0.25,
    figsize: Tuple[int, int] = (12, 8),
    show_labels: bool = True,
    use_numeric_labels: bool = False,
    show_conf: bool = True,
    font_size: int = 10
) -> plt.Figure:
    """Visualize model predictions on the image.
    
    Parameters
    ----------
    image : numpy.ndarray
        Input image
    results : ultralytics.engine.results.Results
        Prediction results from YOLO model
    class_names : Dict[int, str]
        Mapping of class indices to names
    conf_threshold : float
        Confidence threshold for predictions
    figsize : Tuple[int, int]
        Figure size for the plot
    show_labels : bool
        Whether to show labels above bounding boxes
    use_numeric_labels : bool
        Whether to use numeric labels instead of class names
    show_conf : bool
        Whether to show confidence scores
    font_size : int
        Font size for labels
        
    Returns
    -------
    plt.Figure
        Matplotlib figure with visualizations
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Display image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Draw predicted boxes
    for idx, box in enumerate(results.boxes, 1):  # Start counting from 1
        if box.conf.item() < conf_threshold:
            continue
            
        # Get coordinates and class
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cls = int(box.cls.item())
        class_name = class_names.get(cls, f"unknown_{cls}")
        
        # Draw rectangle
        rect = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, color='red', linewidth=2)
        ax.add_patch(rect)
        
        # Add label with confidence if enabled
        if show_labels:
            if use_numeric_labels:
                label_text = f"#{idx}"
            else:
                label_text = class_name
                
            if show_conf:
                label_text += f" {conf:.2f}"
            
            ax.text(x1, y1-5, label_text, color='red', fontsize=font_size, 
                    backgroundcolor='white', bbox=dict(facecolor='white', alpha=0.8))
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    
    return fig

def main():
    st.set_page_config(
        page_title="YOLO Real-Time Inference",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("YOLO Real-Time Object Detection")
    st.markdown("Upload an image and see the model predictions!")
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model path selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="/teamspace/studios/this_studio/dsim/best.pt", # TODO: change this to the path of the model
        help="Path to the YOLO model weights"
    )
    
    # Dataset YAML path
    dataset_yaml = st.sidebar.text_input(
        "Dataset YAML",
        value="Data/3_WebUI_7k/yolo_dataset/dataset.yaml",
        help="Path to the dataset YAML file"
    )
    
    # Visualization settings
    st.sidebar.header("Visualization Settings")
    
    # Show/hide labels toggle
    show_labels = st.sidebar.toggle(
        "Show Labels",
        value=True,
        help="Toggle to show/hide labels above bounding boxes"
    )
    
    # Use numeric labels toggle
    use_numeric_labels = st.sidebar.toggle(
        "Use Numeric Labels",
        value=False,
        help="Toggle to use numeric labels (#1, #2, etc.) instead of class names"
    )
    
    # Show/hide confidence scores
    show_conf = st.sidebar.toggle(
        "Show Confidence Scores",
        value=True,
        help="Toggle to show/hide confidence scores in labels"
    )
    
    # Font size slider
    font_size = st.sidebar.slider(
        "Label Font Size",
        min_value=8,
        max_value=24,
        value=10,
        help="Adjust the font size of labels"
    )
    
    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        help="Minimum confidence score for predictions"
    )
    
    # Initialize session state for model
    if 'model' not in st.session_state:
        st.session_state.model = None
        st.session_state.class_names = None
    
    # Load model and class names
    try:
        if st.session_state.model is None:
            with st.spinner("Loading model..."):
                st.session_state.model = YOLO(model_path)
                st.session_state.class_names = load_class_names(dataset_yaml)
            st.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return
    
    # File uploader
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        try:
            # Read and process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Run inference
            with st.spinner("Running inference..."):
                results = st.session_state.model(image)[0]
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Display predictions
            with col2:
                st.subheader("Model Predictions")
                fig = visualize_predictions(
                    image,
                    results,
                    st.session_state.class_names,
                    conf_threshold,
                    show_labels=show_labels,
                    use_numeric_labels=use_numeric_labels,
                    show_conf=show_conf,
                    font_size=font_size
                )
                st.pyplot(fig)
            
            # Display detection information
            st.subheader("Detection Details")
            detections = []
            for idx, box in enumerate(results.boxes, 1):  # Start counting from 1
                if box.conf.item() >= conf_threshold:
                    cls = int(box.cls.item())
                    class_name = st.session_state.class_names.get(cls, f"unknown_{cls}")
                    conf = box.conf.item()
                    detections.append({
                        "ID": f"#{idx}",
                        "Class": class_name,
                        "Confidence": f"{conf:.2%}"
                    })
            
            if detections:
                st.table(detections)
            else:
                st.info("No objects detected above the confidence threshold.")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main() 