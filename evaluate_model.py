import torch
import torch.nn as nn
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from src import config, dataset, model, utils
import numpy as np

def evaluate():
    print("Loading Test Data...")
    _, test_loader, classes = dataset.get_dataloaders()
    
    if test_loader is None:
        print("Error: Dataset not found.")
        return

    print("Loading Model...")
    net = model.build_model(num_classes=len(classes))
    net.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    net = net.to(config.DEVICE)
    net.eval()
    
    all_preds = []
    all_labels = []
    
    print("Running Evaluation...")
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs = inputs.to(config.DEVICE)
            outputs = net(inputs)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            
    # Metrics
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=classes))
    
    # Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved to confusion_matrix.png")

if __name__ == "__main__":
    evaluate()
