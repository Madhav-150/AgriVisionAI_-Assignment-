import os

# Project Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, 'dataset')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')
MODEL_SAVE_PATH = os.path.join(BASE_DIR, 'best_model.pth')

# Hyperparameters
BATCH_SIZE = 32      # Kept small for CPU
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
IMG_SIZE = 224
NUM_CLASSES = 2
CLASSES = ['diseased', 'healthy']
DEVICE = 'cpu'       # Explicitly setting to CPU as requested, though 'cuda' checked dynamically usually better.
