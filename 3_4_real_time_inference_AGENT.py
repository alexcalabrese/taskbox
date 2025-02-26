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
import os
from dotenv import load_dotenv
try:
    import google.generativeai as genai
except ImportError:
    raise ImportError(
        "Please install google-generativeai package: pip install google-generativeai"
    )

from PIL import Image
import io

# Disable all warnings
warnings.filterwarnings('ignore')
logging.getLogger('ultralytics').setLevel(logging.ERROR)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

load_dotenv()
# Configure Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("Please set GEMINI_API_KEY environment variable")
    
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-2.0-flash-thinking-exp')
# model = genai.GenerativeModel('gemini-exp-1206')

def load_class_names(dataset_yaml: str = "Data/3_WebUI_7k/yolo_dataset/dataset.yaml") -> Dict[int, str]:
    """Load class names from dataset yaml file."""
    with open(dataset_yaml, 'r') as f:
        data = yaml.safe_load(f)
    return data.get('names', {})

def get_box_description(box_info: Dict) -> str:
    """Generate a description of a bounding box and its contents."""
    return f"Box #{box_info['id']}: {box_info['class']} (confidence: {box_info['confidence']:.2f}) at position {box_info['coords']}"

def create_gemini_prompt(task: str, boxes_info: List[Dict], previous_context: str = "") -> str:
    """Create a detailed prompt for Gemini to analyze boxes based on the task."""
    prompt = f"""Task to complete: "{task}"

Available UI elements detected in the image (Bounding Boxes):
{chr(10).join(get_box_description(box) for box in boxes_info)}

{f'Previous actions performed: {previous_context}' if previous_context else ''}

Analysis Instructions:
1. First, understand what UI element we need to interact with based on the task
2. Look for the element in the detected boxes
3. Consider these possible actions:
   - Direct Click: If the target element is clearly visible and detected, select its box number
   - Zoom Required: If the target element is:
     * Too small to be detected accurately
     * Part of a cluster of elements
     * Near detected elements but not detected itself
   - Text Input: For search bars or text fields that need keyboard input

Response Format Options:
1. For Direct Click:
   'Box #X - This element matches the task because [reason]'

2. For Zoom Request:
   'Zoom in on box #X with [25x/50x/100x] zoom - This area likely contains [target element] because [reason]'

3. For Text Input:
   'Box #X - Input "[text]" into this field because [reason]'

4. If No Match Found:
   'No suitable box - [explanation why and suggestion for next step]'

Important Considerations:
- YOLO detection may not be 100% accurate - trust the visual context over the predicted class names
- Consider the UI hierarchy and common web patterns when suggesting interactions
- If the target element isn't detected but you see nearby elements, recommend zooming into that area
- For complex tasks, break them down into sequential steps
- Consider the previous context to maintain task continuity

Please analyze the task and detected elements carefully, then provide a single, clear response following one of the formats above."""
    return prompt

def analyze_task_with_gemini(
    image: np.ndarray,
    image2: np.ndarray,
    boxes_info: List[Dict],
    task: str,
    previous_context: str = ""
) -> Tuple[Optional[int], str]:
    """Use Gemini to analyze which box best matches the task."""
    # Convert image to PIL
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    pil_image2 = Image.fromarray(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    # Create prompt
    prompt = create_gemini_prompt(task, boxes_info, previous_context)
    
    try:
        # Get response from Gemini
        response = model.generate_content([prompt, pil_image, pil_image2])
        response_text = response.text
        
        # Parse response
        if "No suitable box" in response_text:
            return None, response_text
        
        # Extract box number from response (format: "Box #X - explanation")
        box_num = int(response_text.split("#")[1].split(" ")[0])
        return box_num, response_text
        
    except Exception as e:
        logger.error(f"Error getting Gemini response: {str(e)}")
        return None, f"Error: {str(e)}"

def visualize_predictions(
    image,
    results,
    class_names: Dict[int, str],
    conf_threshold: float = 0.25,
    figsize: Tuple[int, int] = (12, 8),
    show_labels: bool = True,
    use_numeric_labels: bool = False,
    show_conf: bool = True,
    font_size: int = 10,
    selected_box: Optional[int] = None,
    box_opacity: float = 0.3,
    label_opacity: float = 0.8
) -> Tuple[plt.Figure, List[Dict], np.ndarray]:
    """Visualize model predictions on the image."""
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create a copy of the image for drawing boxes
    image_with_boxes = image.copy()
    
    # Display image
    ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Store box information for Gemini
    boxes_info = []
    
    # Draw predicted boxes
    for idx, box in enumerate(results.boxes, 1):  # Start counting from 1
        if box.conf.item() < conf_threshold:
            continue
            
        # Get coordinates and class
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        conf = box.conf.item()
        cls = int(box.cls.item())
        class_name = class_names.get(cls, f"unknown_{cls}")
        
        # Store box information
        boxes_info.append({
            "id": idx,
            "class": class_name,
            "confidence": conf,
            "coords": [x1, y1, x2, y2]
        })
        
        # Determine box color (green if selected, red otherwise)
        box_color = 'green' if idx == selected_box else 'red'
        box_color_bgr = (0, 255, 0) if idx == selected_box else (0, 0, 255)
        
        # Draw on matplotlib figure
        rect_fill = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=True, 
                                color=box_color, alpha=box_opacity)
        rect_border = plt.Rectangle((x1, y1), x2-x1, y2-y1, fill=False, 
                                  color=box_color, linewidth=2)
        ax.add_patch(rect_fill)
        ax.add_patch(rect_border)
        
        # Draw on image copy
        cv2.rectangle(image_with_boxes, (int(x1), int(y1)), (int(x2), int(y2)), 
                     box_color_bgr, 2)
        
        # Add label with confidence if enabled
        if show_labels:
            if use_numeric_labels:
                label_text = f"#{idx}"
            else:
                label_text = class_name
                
            if show_conf:
                label_text += f" {conf:.2f}"
            
            # Add label to matplotlib figure    
            ax.text(x1, y1-5, label_text, color=box_color, fontsize=font_size, 
                    bbox=dict(facecolor='white', alpha=label_opacity, edgecolor=box_color, pad=2))
            
            # Add label to image copy
            cv2.putText(image_with_boxes, label_text, (int(x1), int(y1)-5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color_bgr, 2)
    
    # Remove axes
    ax.axis('off')
    plt.tight_layout()
    
    return fig, boxes_info, image_with_boxes

def main():
    st.set_page_config(
        page_title="YOLO Real-Time Inference with Gemini",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("YOLO Real-Time Object Detection with Gemini")
    st.markdown("Upload an image and specify a task to find relevant UI elements!")
    
    # Initialize session state
    if 'task_history' not in st.session_state:
        st.session_state.task_history = []
    if 'current_task' not in st.session_state:
        st.session_state.current_task = None
    
    # Sidebar configuration
    st.sidebar.header("Model Configuration")
    
    # Model path selection
    model_path = st.sidebar.text_input(
        "Model Path",
        value="/teamspace/studios/this_studio/dsim/best.pt",
        help="Path to the YOLO model weights"
    )
    
    # Dataset YAML path
    dataset_yaml = st.sidebar.text_input(
        "Dataset YAML",
        value="Data/3_WebUI_7k/yolo_dataset/dataset.yaml",
        help="Path to the dataset YAML file"
    )
    
    # Task input
    task = st.text_input(
        "Specify your task",
        help="Describe what you want to do (e.g., 'Click the submit button')"
    )
    
    # New task button
    if st.button("Start New Task"):
        st.session_state.task_history = []
        st.session_state.current_task = None
    
    # Visualization settings
    st.sidebar.header("Visualization Settings")
    
    # Basic settings
    show_labels = st.sidebar.toggle("Show Labels", value=True)
    use_numeric_labels = st.sidebar.toggle("Use Numeric Labels", value=False)
    show_conf = st.sidebar.toggle("Show Confidence Scores", value=True)
    
    # Advanced visualization settings
    st.sidebar.subheader("Advanced Settings")
    font_size = st.sidebar.slider("Label Font Size", min_value=8, max_value=24, value=10)
    box_opacity = st.sidebar.slider("Box Opacity", min_value=0.0, max_value=1.0, value=0.3, step=0.1)
    label_opacity = st.sidebar.slider("Label Background Opacity", min_value=0.0, max_value=1.0, value=0.8, step=0.1)
    conf_threshold = st.sidebar.slider("Confidence Threshold", min_value=0.0, max_value=1.0, value=0.25)
    
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
    
    if uploaded_file is not None and task:
        try:
            # Read and process image
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            
            # Run inference
            with st.spinner("Running inference..."):
                results = st.session_state.model(image)[0]
            
            # Update task history if new task
            if task != st.session_state.current_task:
                st.session_state.current_task = task
                st.session_state.task_history.append(task)
            
            # Create two columns
            col1, col2 = st.columns(2)
            
            # Get previous context
            previous_context = "\n".join([
                f"Previous task {i+1}: {t}" 
                for i, t in enumerate(st.session_state.task_history[:-1])
            ])
            
            # Analyze with Gemini
            with st.spinner("Analyzing task with Gemini..."):
                fig, boxes_info , image_with_boxes= visualize_predictions(
                    image,
                    results,
                    st.session_state.class_names,
                    conf_threshold,
                    show_labels=show_labels,
                    use_numeric_labels=use_numeric_labels,
                    show_conf=show_conf,
                    font_size=font_size
                )
                
                selected_box, explanation = analyze_task_with_gemini(
                    image, 
                    image_with_boxes,
                    boxes_info, 
                    task,
                    previous_context
                )
            
            # Display original image
            with col1:
                st.subheader("Original Image")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            
            # Display predictions
            with col2:
                st.subheader("Model Predictions")
                fig, _ ,_= visualize_predictions(
                    image,
                    results,
                    st.session_state.class_names,
                    conf_threshold,
                    show_labels=show_labels,
                    use_numeric_labels=use_numeric_labels,
                    show_conf=show_conf,
                    font_size=font_size,
                    selected_box=selected_box
                )
                st.pyplot(fig)
            
            # Display Gemini's analysis
            st.subheader("Task Analysis")
            st.write(explanation)
            
            # Display detection information
            st.subheader("Detection Details")
            detections = []
            for idx, box in enumerate(results.boxes, 1):
                if box.conf.item() >= conf_threshold:
                    cls = int(box.cls.item())
                    class_name = st.session_state.class_names.get(cls, f"unknown_{cls}")
                    conf = box.conf.item()
                    detections.append({
                        "ID": f"#{idx}",
                        "Class": class_name,
                        "Confidence": f"{conf:.2%}",
                        "Selected": "✓" if idx == selected_box else ""
                    })
            
            if detections:
                st.table(detections)
            else:
                st.info("No objects detected above the confidence threshold.")
            
        except Exception as e:
            st.error(f"Error processing image: {str(e)}")

if __name__ == "__main__":
    main()
    
    
# to run the code: streamlit run 3_4_real_time_inference_AGENT.py