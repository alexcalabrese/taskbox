# UI Element Detection and Classification

A comprehensive deep learning project for detecting and classifying UI elements in mobile and web applications using YOLO (You Only Look Once), various classification models, and Gemini AI for intelligent interaction.

## Demo

https://github.com/user-attachments/assets/d116599f-4f1d-44fe-a3c3-c7fa1e051bcb

## Project Overview

This project implements a multi-stage approach to UI element detection and classification:

1. **Data Processing and Preparation**
   - Download and format datasets for training
   - Automatic YOLO format conversion for bounding boxes

2. **Model Training**
   - YOLO-based object detection
   - Binary classification models
   - Multi-class classification models
   - CLIP and CNN-based approaches

3. **Inference**
   - Real-time inference capabilities
   - Support for different input sources
   - Intelligent agent-based inference system with Gemini AI integration
   - Interactive web interface using Streamlit


Watch our intelligent agent system in action! The demo shows:
1. Upload of a UI screenshot
2. Natural language task specification
3. Real-time element detection
4. Intelligent element selection
5. Task analysis and recommendations

Try it yourself:
```bash
# Start the Streamlit interface
streamlit run 3_4_real_time_inference_AGENT.py

# Then:
1. Upload any UI screenshot
2. Enter a task (e.g., "Click the submit button")
3. Watch as the system analyzes and suggests the best action
```

Example tasks to try:
- "Click the login button"
- "Find the search box and enter 'products'"
- "Select the dropdown menu"
- "Click the close icon in the top right"

The system will intelligently:
- Identify relevant UI elements
- Suggest appropriate actions
- Handle complex multi-step tasks
- Provide clear visual feedback

## Environment Setup

1. Install the required dependencies:
```bash
pip install -r requirements.txt
```

2. Set up your Gemini API key:
```bash
# Create a .env file and add your Gemini API key
echo "GEMINI_API_KEY=your_api_key_here" > .env
```

## Project Structure

```
.
├── Data Processing
│   ├── 2_0_download_data.py
│   ├── 2_1_create_dataframe.py
│   └── 3_0_download_and_format_dataset.py
│
├── Binary Classification
│   ├── 2_2_prepare_binary_classification.py
│   ├── 2_3_train_binary_classification.py
│   ├── 2_4_train_binary_classification_clip.py
│   └── 2_5_train_binary_classification_cnn.py
│
├── Multiclass Classification
│   ├── 2_6_prepare_multiclass_classification.py
│   ├── 2_7_train_multiclass_classification.py
│   ├── 2_8_train_multiclass_classification_clip.py
│   └── 2_9_train_multiclass_classification_cnn.py
│
├── YOLO Training and Inference
│   ├── 3_1_train_yolo.py
│   ├── 3_2_inference.py
│   ├── 3_3_real_time_inference.py
│   └── 3_4_real_time_inference_AGENT.py
│
└── Notebooks
    ├── 3_0_explore_dataset.ipynb
    └── 3_5_gemini_second_dataset.ipynb
```

## Usage

### 1. Data Preparation

```bash
# Download and prepare the dataset
python 3_0_download_and_format_dataset.py

# Create dataframes for training
python 2_1_create_dataframe.py
```

### 2. Training Models

```bash
# Train YOLO model
python 3_1_train_yolo.py --model_size n --epochs 100 --batch_size 16

# Train binary classification
python 2_3_train_binary_classification.py

# Train multiclass classification
python 2_7_train_multiclass_classification.py
```

### 3. Inference

```bash
# Regular inference
python 3_2_inference.py --model_path path/to/model --source path/to/images

# Real-time inference
python 3_3_real_time_inference.py --model_path path/to/model

# Interactive Agent-based inference with Streamlit UI
streamlit run 3_4_real_time_inference_AGENT.py
```

## Intelligent Agent System

The project includes an advanced agent-based inference system (`3_4_real_time_inference_AGENT.py`) that combines YOLO object detection with Google's Gemini AI for intelligent UI interaction:

### Features

1. **Interactive Web Interface**
   - Built with Streamlit for easy interaction
   - Real-time visualization of detection results
   - Configurable visualization settings
   - Task history tracking

2. **Intelligent Task Analysis**
   - Natural language task processing
   - Context-aware element selection
   - Multi-step task planning
   - Confidence-based decision making

3. **Advanced Visualization**
   - Customizable bounding box display
   - Adjustable confidence thresholds
   - Numeric or class-based labels
   - Selected element highlighting

4. **Action Types**
   - Direct element clicks
   - Zoom recommendations for small/clustered elements
   - Text input suggestions
   - Multi-step interaction planning

### Configuration Options

- Model path and dataset YAML configuration
- Visualization settings:
  - Label display options
  - Box and label opacity
  - Font size customization
  - Confidence thresholds
- Task history management
- Real-time analysis updates

## Model Architectures

1. **YOLO Detection**
   - YOLOv8 architecture
   - Multiple model sizes (nano to xlarge)
   - Custom-trained on UI element dataset

2. **Classification Models**
   - Binary classification for element detection
   - Multi-class classification for element type identification
   - CLIP-based models for zero-shot learning
   - Custom CNN architectures

3. **Gemini AI Integration**
   - Task understanding and decomposition
   - Context-aware element selection
   - Natural language interaction
   - Multi-modal analysis (image and text)


## Contributing

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

