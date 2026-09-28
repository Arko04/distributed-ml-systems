#https://gemini.google.com/share/787646786e71
#https://gemini.google.com/share/d4c36410890d


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import sys

path = '../Data/'
# --- 1. Data Loading ---
# Load the three separate datasets from their .npy files
print("Loading data from .npy files...")
try:
    # Define file paths
    path_X1 = path + 'data1.npy'
    path_y1 = path + 'labels1.npy'
    path_X2 = path + 'data2.npy'
    path_y2 = path + 'labels2.npy'
    path_X3 = path + 'data3.npy'
    path_y3 = path + 'labels3.npy'

    # Load data
    X1 = np.load(path_X1)
    y1 = np.load(path_y1)
    X2 = np.load(path_X2)
    y2 = np.load(path_y2)
    X3 = np.load(path_X3)
    y3 = np.load(path_y3)
    
    print("All 3 datasets loaded successfully.")

except FileNotFoundError as e:
    print(f"Error: Could not find file {e.filename}", file=sys.stderr)
    print("Please make sure the file paths are correct and accessible.", file=sys.stderr)
    sys.exit(1) # Exit the script if data can't be loaded

# --- 2. Data Preparation ---
# 2a. Merge all three datasets as required 
X_all = np.concatenate((X1, X2, X3), axis=0)
y_all = np.concatenate((y1, y2, y3), axis=0)
print(f"Total merged dataset shape: X={X_all.shape}, y={y_all.shape}")

unique_labels = np.unique(y_all)
# Print unique labels
print(f"unique labels: {unique_labels}")

# 2b. Split into 80% train and 20% test 
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, random_state=42
)

# 2c. Standardize the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 2d. Convert to PyTorch Tensors
# Assuming y data is already 0 or 1
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

print(f"Training set size: {X_train_tensor.shape[0]}")
print(f"Testing set size:  {X_test_tensor.shape[0]}")

# --- 3. Model Definition ---
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# --- 4. Training Process ---

# Hyperparameters from the prompt
learning_rate = 0.01 
num_epochs = 30        
n_samples, n_features = X_train.shape

# Initialize the model
model = LogisticRegressionModel(n_features)


# Requirement: Initialize weights to zero vector
# without training the model and tracking the gradients
with torch.no_grad():
    model.linear.weight.fill_(0.0)
    model.linear.bias.fill_(0.0)

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

print(f"\nStarting serial training for {num_epochs} epochs...")
# Start the timer for training 
start_time = time.time()

# Training loop
for epoch in range(num_epochs):
    # Forward pass
    y_pred = model(X_train_tensor)
    
    # Compute loss
    loss = criterion(y_pred, y_train_tensor)
    
    # Backward pass and optimize
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if (epoch + 1) % 5 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Stop the timer
end_time = time.time()
total_training_time = end_time - start_time

print("Training finished.")

# --- 5. Evaluation and Reporting ---

# Set model to evaluation mode
model.eval()

with torch.no_grad():
    # Get predictions on the test set
    y_pred_probs = model(X_test_tensor)
    
    # Convert probabilities to binary class (0 or 1)
    y_pred_class = y_pred_probs.round()
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_class.numpy())

print("\n--- 📊 Baseline Serial Results ---")
print(f"Total Training Time: {total_training_time:.4f} seconds")
print(f"Final Test Accuracy: {accuracy * 100:.2f}%")