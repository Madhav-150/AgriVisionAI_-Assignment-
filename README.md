# AgriVision AI - Leaf Disease Classification

This project is an end-to-end AI/ML pipeline for classifying agricultural leaf images into **Healthy** or **Diseased** categories. It uses a **MobileNetV2** transfer learning model optimized for CPU inference and includes a Streamlit-based web interface.

## 📁 Project Structure
```
AgriVision_AI/
├── dataset/             # (User to create)
│   ├── train/
│   │   ├── healthy/
│   │   └── diseased/
│   └── test/
│       ├── healthy/
│       └── diseased/
├── src/                 # Core Modules
│   ├── config.py        # Settings (Batch size, paths)
│   ├── dataset.py       # Data loading & augmentation
│   ├── model.py         # MobileNetV2 definition
│   └── utils.py         # Helpers
├── data_pipeline.py     # Verify data loading
├── model_train.py       # Training script
├── evaluate_model.py    # Evaluation metrics
├── predict.py           # CLI Inference
├── app.py               # Streamlit Dashboard
└── requirements.txt     # Dependencies
```

## 🚀 Setup Instructions

1.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```


2.  **Dataset Preparation**
    *   **Download**: You have the `New Plant Diseases Dataset`.
    *   **Strategy**: We will use **Tomato** images for this assignment.
    *   **Mapping**: Copy images from your downloaded folders to the project folders as follows:



    | Destination Folder (AgriVision_AI) | Source Folder (Downloaded Dataset) | Instructions |
    | :--- | :--- | :--- |
    | `dataset/train/healthy` | `Train/Tomato___healthy` | Copy **200** images. |
    | `dataset/train/diseased` | `Train/Tomato___Bacterial_spot` | Copy **200** images. |
    | `dataset/test/healthy` | `Valid/Tomato___healthy` | Copy **50** images. |
    | `dataset/test/diseased` | `Valid/Tomato___Bacterial_spot` | Copy **50** images. |

    *   *Note: Using the 'Valid' folder for testing is perfect as it ensures the model hasn't seen these images during training.*


## 🛠️ Usage

### 1. Train the Model
Train the Transfer Learning model (MobileNetV2) on your dataset.
```bash
python model_train.py
```
*   Config: 5 Epochs, Adam Optimizer, CrossEntropy Loss.
*   Saves the best model to `best_model.pth`.

### 2. Verify Data Pipeline
Visualize how images are resized and augmented.
```bash
python data_pipeline.py
```

### 3. Evaluate Performance
Generate classification report and confusion matrix on the test set.
```bash
python evaluate_model.py
```

### 4. Run CLI Inference
Predict a single image from the command line.
```bash
python predict.py --image "path/to/leaf.jpg"
```

### 5. Launch Web App
Start the interactive dashboard for real-time analysis.
```bash
streamlit run app.py
```

## ⏱️ Performance Note (CPU)
*   **MobileNetV2** was chosen for its efficiency.
*   Estimated training time on CPU for full dataset (50k images): **3+ hours/epoch**.
*   Estimated training time for subset (1k images): **~5-10 minutes**.

## 📊 Approach
*   **Preprocessing**: Resizing to 224x224, Random Floips/Rotations, Normalization.
*   **Model**: Freeze MobileNetV2 features, fine-tune custom classifier head.
*   **Metric**: Accuracy, Precision, Recall, F1-Score.

## 📈 Results
The model achieved **~99% accuracy** on the test dataset.

### Visual Outputs
Below are examples of the model's predictions:

![Output 1](AgriVision_Output(1).jpeg)
*(Caption: Model Prediction on Healthy Leaf)*

![Output 2](AgriVision_Output(2).jpeg)
*(Caption: Prediction on Diseased Leaf)*

![Output 3](AgriVision_Output(3).jpeg)
*(Caption: App Interface - Healthy)*

![Output 4](AgriVision_Output(4).jpeg)
*(Caption: App Interface - Diseased)*

---

### ⚠️ Note on Edge Deployment (TFLite)
The project includes optional scripts (`export_tflite.py` and `tflite_infer.py`) for converting the trained model to **TensorFlow Lite** for mobile/edge deployment. 

*   **Status**: Scripts are code-complete and included in the repository.
*   **Constraint**: These files require a specific Python environment (e.g., Python 3.10) to run. 
*   **Observation**: We attempted to generate the model using **Google Colab**, but encountered version 
     incompatibility issues even in that environment.
*   **Solution**: Consequently, we opted to host the project on **Streamlit Cloud** as the primary 
     deployment method, which works flawlessly.




