import numpy as np
import os
import argparse
from PIL import Image

# Try importing TFLite runtime or TensorFlow
try:
    import tflite_runtime.interpreter as tflite
except ImportError:
    try:
        import tensorflow.lite as tflite
    except ImportError:
        tflite = None

from src import config

def load_image(image_path):
    img = Image.open(image_path).convert('RGB')
    img = img.resize((config.IMG_SIZE, config.IMG_SIZE))
    img_array = np.array(img, dtype=np.float32)
    
    # Normalize (standard ImageNet normalization)
    # mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    img_array /= 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    img_array = (img_array - mean) / std
    
    # Add batch dimension (1, 224, 224, 3) -> TFLite usually expects NHWC or NCHW depending on conversion
    # PyTorch is NCHW (1, 3, 224, 224).
    # If converted via ONNX->TF, it might preserve NCHW or convert to NHWC.
    # Usually standard TF conversion handles this via transposes.
    # Let's check the input shape from the interpreter.
    
    return img_array

def run_inference(model_path, image_path):
    if tflite is None:
        print("Error: neither 'tensorflow' nor 'tflite_runtime' is installed.")
        print("Please install them to run this script.")
        return

    # Load the TFLite model and allocate tensors.
    if hasattr(tflite, 'Interpreter'):
        interpreter = tflite.Interpreter(model_path=model_path)
    else:
         interpreter = tflite.Interpreter(model_path=model_path)

    interpreter.allocate_tensors()

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Check expected shape
    input_shape = input_details[0]['shape']
    print(f"Model expects input shape: {input_shape}")
    
    # Preprocess image
    input_data = load_image(image_path)
    
    # Handle NCHW vs NHWC match
    # input_data is currently HWC (224, 224, 3) normalized
    # PyTorch export usually expects (1, 3, 224, 224)
    # But if converted to TF, TF loves (1, 224, 224, 3)
    
    if input_shape[1] == 3: # NCHW
        input_data = input_data.transpose((2, 0, 1)) # HWC -> CHW
        input_data = np.expand_dims(input_data, axis=0)
    else: # NHWC
        input_data = np.expand_dims(input_data, axis=0) # Add batch dim
    
    # Run Inference
    interpreter.set_tensor(input_details[0]['index'], input_data)
    interpreter.invoke()

    # Get output
    output_data = interpreter.get_tensor(output_details[0]['index'])
    print(f"Raw Output: {output_data}")
    
    # Softmax
    def softmax(x):
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()
        
    probs = softmax(output_data[0])
    class_idx = np.argmax(probs)
    class_name = config.CLASSES[class_idx]
    confidence = probs[class_idx]
    
    print(f"\nResult: {class_name} ({confidence*100:.2f}%)")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--model", type=str, default="model.tflite", help="Path to .tflite model")
    args = parser.parse_args()
    
    if not os.path.exists(args.model):
        print(f"Model file {args.model} not found.")
    else:
        run_inference(args.model, args.image)
