import torch
import argparse
from PIL import Image
from src import config, dataset, model

def predict_image(image_path):
    # 1. Load Model
    # Note: We need to know classes. If config.CLASSES is fixed, use that.
    classes = config.CLASSES 
    
    net = model.build_model(num_classes=len(classes))
    try:
        net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    except FileNotFoundError:
        print("Error: Model file not found. Please train the model first.")
        return

    net = net.to(config.DEVICE)
    net.eval()
    
    # 2. Preprocess Image
    transforms = dataset.get_transforms('test')
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"Error calling Image.open: {e}")
        return

    img_t = transforms(img).unsqueeze(0).to(config.DEVICE)
    
    # 3. Predict
    with torch.no_grad():
        outputs = net(img_t)
        probs = torch.nn.functional.softmax(outputs, dim=1)
        conf, predicted = torch.max(probs, 1)
        
    class_name = classes[predicted.item()]
    confidence = conf.item() * 100
    
    print(f"Prediction: {class_name}")
    print(f"Confidence: {confidence:.1f}%")
    return class_name, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='AgriVision AI Inference')
    parser.add_argument('--image', type=str, required=True, help='Path to input image')
    args = parser.parse_args()
    
    predict_image(args.image)
