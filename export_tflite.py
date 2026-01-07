import torch
import torch.nn as nn
from src import config, model
import os
import onnx

def export_model():
    print("Loading PyTorch model...")
    # 1. Load PyTorch Model
    classes = config.CLASSES
    net = model.build_model(num_classes=len(classes), pretrained=False) # No need to download weights again, loading state_dict
    
    try:
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location='cpu'))
    except FileNotFoundError:
        print(f"Error: Model not found at {config.MODEL_SAVE_PATH}")
        return

    net.eval()
    
    # 2. Export to ONNX
    print("Exporting to ONNX...")
    dummy_input = torch.randn(1, 3, config.IMG_SIZE, config.IMG_SIZE)
    onnx_path = os.path.join(config.BASE_DIR, "model.onnx")
    
    torch.onnx.export(
        net, 
        dummy_input, 
        onnx_path,
        export_params=True,
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"ONNX model saved to {onnx_path}")
    
    
    # 3. Export to TF/TFLite
    try:
        import tensorflow as tf
        from onnx_tf.backend import prepare
        
        print("Converting ONNX to TensorFlow/TFLite...")
        # Load ONNX model
        onnx_model = onnx.load(onnx_path)
        
        # Convert to TF SavedModel
        tf_model_path = os.path.join(config.BASE_DIR, "model_tf")
        tf_rep = prepare(onnx_model)
        tf_rep.export_graph(tf_model_path)
        print(f"TF SavedModel saved to {tf_model_path}")
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_saved_model(tf_model_path)
        # Optional: Optimizations
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        tflite_path = os.path.join(config.BASE_DIR, "model.tflite")
        with open(tflite_path, "wb") as f:
            f.write(tflite_model)
            
        print(f"Success! TFLite model saved to {tflite_path}")
    except ImportError as e:
        print(f"Warning: Could not export to TFLite because TensorFlow/onnx-tf is not installed or compatible. Error: {e}")
        print("The ONNX model is available and can be used for deployment or converted elsewhere (e.g., Google Colab).")

if __name__ == "__main__":
    export_model()
