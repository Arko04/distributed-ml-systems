#https://gemini.google.com/share/787646786e71
#https://gemini.google.com/share/d4c36410890d
#https://gemini.google.com/share/301837ad94ac
#https://gemini.google.com/share/7aeaed023a28

#my prompts:
# How to get command line arguments in python. add this to the logreg_serial.py code.
# simple example of how to use comm.bcast and comm.gather in mpi4py.
# functinalize the code into functions that do one thing.
# debug the code.
# how to do averaging of model weights in pytorch. give me the code snippet.

from mpi4py import MPI
import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import time
import sys
import argparse  # <-- 1. IMPORTED

# --- 1. Model Definition ---
class LogisticRegressionModel(nn.Module):
    def __init__(self, n_features):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(n_features, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# --- 2. NEW Argument Parsing Function ---
def parse_arguments():
    """
    Parses command-line arguments for num_rounds and num_epochs.
    """
    parser = argparse.ArgumentParser(description="Federated Learning MPI Script")
    parser.add_argument(
        '--num_rounds', 
        type=int, 
        default=10, 
        help="The number of federated training rounds (default: 10)"
    )
    parser.add_argument(
        '--num_epochs', 
        type=int, 
        default=10, 
        help="The number of local training epochs per round (default: 30)"
    )
    args = parser.parse_args()
    return args

# --- 3. Federated Learning Helper Functions ---

def average_state_dicts(state_dicts):
    """
    Averages a list of PyTorch state_dict objects.
    """
    if not state_dicts:
        return None
    
    avg_state_dict = OrderedDict()
    keys = state_dicts[0].keys()
    
    for key in keys:
        stacked_tensors = torch.stack([sd[key].float() for sd in state_dicts])
        avg_tensor = torch.mean(stacked_tensors, dim=0)
        avg_state_dict[key] = avg_tensor
        
    return avg_state_dict

def train_model(model, criterion, optimizer, X_train_tensor, y_train_tensor, local_epochs):
    """
    Trains the model locally on the worker's data.
    """
    for epoch in range(local_epochs):
        y_pred = model(X_train_tensor)
        loss = criterion(y_pred, y_train_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# --- 4. Data Processing Helper Functions ---

def load_numpy_data(rank, data_path):
    """
    Function's "one thing": Load X and y data from .npy files for a worker.
    """
    path_x = f"{data_path}data{rank}.npy"
    path_y = f"{data_path}labels{rank}.npy"

    try:
        X = np.load(path_x)
        y = np.load(path_y)
        return X, y
    except FileNotFoundError:
        print(f"Error: Data files not found for rank {rank} at path {data_path}")
        return None, None

def scale_data(X_train, X_test):
    """
    Function's "one thing": Scale training and testing data.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def create_tensors(X_train, y_train, X_test, y_test):
    """
    Function's "one thing": Convert NumPy arrays to PyTorch tensors.
    """
    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    y_test_tensor = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)
    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor

# --- 5. Model Setup & Evaluation Functions ---

def initialize_weights(model):
    """
    Function's "one thing": Set model weights and bias to zero.
    """
    with torch.no_grad():
        model.linear.weight.fill_(0.0)
        model.linear.bias.fill_(0.0)

def evaluate_model(model, X_test_tensor, y_test_tensor, total_training_time, accuracy):
    """
    Function's "one thing": Evaluate the model and print final results.
    """
    model.eval()
    with torch.no_grad():
        y_pred_probs = model(X_test_tensor)
        y_pred_class = y_pred_probs.round()
        accuracy = accuracy_score(y_test_tensor.numpy(), y_pred_class.numpy())

    print(f"\n--- 📊 [Rank {rank}] Final Results ---")
    print(f"Total Training Time: {total_training_time:.4f} seconds")
    print(f"Final Test Accuracy: {accuracy * 100:.2f}%")
    return accuracy

# --- 6. Main MPI Orchestration ---

def run_federated_training(comm, rank, model, criterion, optimizer, 
                           X_train_tensor, y_train_tensor, 
                           num_rounds, num_epochs):
    """
    Function's "one thing": Manage the main federated training and averaging loop.
    """
    for i in range(num_rounds):
        if rank != 0:
            train_model(model, criterion, optimizer, 
                        X_train_tensor, y_train_tensor, 
                        num_epochs)

        weights = model.state_dict()
        weights_list = comm.gather(weights, root=0)
        
        avg_state_dict = None
        if rank == 0:
            # print(f"\n--- MASTER: Round {i+1}/{num_rounds} ---")
            worker_weights_list = weights_list[1:] 
            avg_state_dict = average_state_dicts(worker_weights_list)
            # print("MASTER: Averaging complete. Broadcasting new model.")
        
        updated_state_dict = comm.bcast(avg_state_dict, root=0)
        model.load_state_dict(updated_state_dict)
        comm.Barrier()

    print(f"--- [Rank {rank}] Federated Training Complete ---")


# --- 7. Main Execution ---

def main():
    # --- Parse Arguments ---
    args = parse_arguments() # <-- 2. CALLED THE FUNCTION
    
    # --- Hyperparameters ---
    path = '../Data/'
    learning_rate = 0.01 
    n_features = 50
    # Use parsed arguments instead of hardcoded values
    num_rounds = args.num_rounds # <-- 3. USED THE ARGS
    num_epochs = args.num_epochs # <-- 3. USED THE ARGS

    # --- MPI Setup ---
    global comm, rank, size # Make MPI info global for helper functions
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if rank == 0:
        print(f"--- Starting Federated Learning ---")
        print(f"Total Processes: {size}")
        print(f"Number of Rounds: {num_rounds}")
        print(f"Local Epochs: {num_epochs}")
        print("-----------------------------------")


    # --- Model Initialization (All Processes) ---
    model = LogisticRegressionModel(n_features)
    initialize_weights(model) 
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # --- Data Pipeline (Workers Only) ---
    if rank != 0:
        X, y = load_numpy_data(rank, path)
        if X is None:
            comm.Abort() 

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )
        # X_train_scaled, X_test_scaled = scale_data(X_train, X_test)
        X_train_scaled, X_test_scaled = X_train, X_test
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = create_tensors(
            X_train_scaled, y_train, X_test_scaled, y_test
        )
    else:
        X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor = (None, None, None, None)

    # --- Run Federated Training ---
    start_time = time.time()

    run_federated_training(comm, rank, model, criterion, optimizer, 
                           X_train_tensor, y_train_tensor, 
                           num_rounds, num_epochs) # Pass args here
                           
    total_training_time = time.time() - start_time

    # --- Final Evaluation (Workers Only) ---
    accuracy = 0.0
    if rank != 0:
        # evaluate_model(model, X_test_tensor, y_test_tensor, total_training_time, accuracy)
        accuracy = evaluate_model(model, X_test_tensor, y_test_tensor, total_training_time, rank)
    else:
        print("\n--- MASTER: Task complete. ---")

    total_accuracy_sum = comm.reduce(accuracy, op=MPI.SUM, root=0)

    if rank==0:
        print("the final average accuracy across workers is:", total_accuracy_sum/(size-1))



if __name__ == "__main__":
    main()