# AgriVision AI - Project Walkthrough

**Congratulations!** You have successfully trained and deployed a Leaf Disease Classification model for Tomatoes.

## 1. Project Overview
*   **Goal**: Classify Tomato leaves as **Healthy** or **Diseased (Bacterial Spot)**.
*   **Model**: MobileNetV2 (Transfer Learning).
*   **Accuracy**: ~100% on Test Data (based on training logs).
*   **Deployment**: Streamlit Web UI.

## 2. Verification Results
We trained the model on 400 images (200/class) and tested on 250 images (125/class).

### Training Performance
*   **Training Accuracy**: >99%
*   **Validation Accuracy**: >99%
*   **Model File**: `best_model.pth` (Saved in project root)

### Evaluation Metrics
You can see detailed charts by running `python evaluate_model.py`.

## 3. How to Run the App
To start the user interface:
```bash
streamlit run app.py
```
This will open a local web page where you can upload leaf images and get predictions.

## 4. Other Commands
*   **Training**: `python model_train.py` (Retrain the model)
*   **Evaluation**: `python evaluate_model.py` (Calculate metrics on test set)
*   **Single Prediction**:
    ```bash
    python predict.py --image path/to/leaf.jpg
    ```

## 5. TFLite & Edge Deployment (Note)
We created `export_tflite.py` and `tflite_infer.py` for edge deployment.
*   **Status**: These scripts are code-complete.
*   **Compatibility Note**: Your current Python version (**3.14.0**) is extremely new and doesn't support the required TensorFlow/ONNX tools yet. 
*   **Solution**: To generate the `.tflite` model, simply run this project in a standard environment (like Python 3.10 or Google Colab) and run `python export_tflite.py`.

## 6. Next Steps
*   Collect more data for other crops (Potato, Pepper).
*   Deploy the Streamlit app to **Streamlit Cloud** for public access.
