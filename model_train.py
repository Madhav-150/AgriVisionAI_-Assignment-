import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from src import config, dataset, model, utils
import os

def train_model():
    utils.seed_everything()
    
    # 1. Load Data
    print("Initializing Data Loaders...")
    train_loader, test_loader, classes = dataset.get_dataloaders()
    
    if train_loader is None:
        print("Error: Dataset not found. Please place images in 'dataset/train' and 'dataset/test'.")
        return

    # 2. Build Model
    print(f"Building Model (Classes: {classes})...")
    net = model.build_model(num_classes=len(classes))
    net = net.to(config.DEVICE)
    
    # 3. Define Loss & Optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)
    
    # 4. Training Loop
    best_acc = 0.0
    history = {'acc': [], 'val_acc': [], 'loss': [], 'val_loss': []}
    
    print(f"Starting training on {config.DEVICE} for {config.NUM_EPOCHS} epochs...")
    
    for epoch in range(config.NUM_EPOCHS):
        net.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Training Phase
        with tqdm(train_loader, unit="batch") as tepoch:
            tepoch.set_description(f"Epoch {epoch+1}/{config.NUM_EPOCHS}")
            
            for inputs, labels in tepoch:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                
                optimizer.zero_grad()
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                tepoch.set_postfix(loss=loss.item(), acc=100. * correct / total)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = correct / total
        
        # Validation Phase
        net.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(config.DEVICE), labels.to(config.DEVICE)
                outputs = net(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
        
        val_loss = val_loss / len(test_loader.dataset)
        val_acc = val_correct / val_total
        
        print(f"Stats: Train Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} | Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}")
        
        # Save History
        history['acc'].append(epoch_acc)
        history['val_acc'].append(val_acc)
        history['loss'].append(epoch_loss)
        history['val_loss'].append(val_loss)
        
        # Checkpoint
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(net.state_dict(), config.MODEL_SAVE_PATH)
            print(f"New best model saved with accuracy: {best_acc:.4f}")

    # Plot results
    utils.plot_training_history(history)
    print("Training complete.")

if __name__ == "__main__":
    train_model()
