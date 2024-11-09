import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import custom modules
from utils.board_repr import board_to_input_tensor
from utils.move_mapping import move_to_index
from utils.constants import INPUT_SIZE, HIDDEN_SIZE, ACTION_SPACE_SIZE
from models.neural import ChessNet
from datasets.chess_dataset import ChessIterableDataset

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
ZIP_PATH = "data/datasets.zip"  # Path to your zip file
CSV_FILENAME = "datasets/tactic_evals.csv"  # Include the subdirectory

MODEL_SAVE_PATH = "saved_models/chess_model.pth"

# Parameters
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories if they don't exist
os.makedirs("saved_models", exist_ok=True)

if __name__ == "__main__":
    # ------------------------
    # Data Loading Section
    # ------------------------

    print("Loading dataset from zip file...")

    # Create the dataset and manually separate into training and validation
    dataset = ChessIterableDataset(ZIP_PATH, CSV_FILENAME, val_split=0.2)

    # Separate training and validation data based on the split flag
    train_data = []
    val_data = []

    for input_tensor, target_index, split in dataset:
        if split == "train":
            train_data.append((input_tensor, target_index))
        else:
            val_data.append((input_tensor, target_index))

    # Create DataLoaders for training and validation sets
    train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False)

    # ------------------------
    # Model Training Section
    # ------------------------

    # Initialize model, loss function, and optimizer
    model = ChessNet(INPUT_SIZE, HIDDEN_SIZE).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        for inputs, targets in train_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            # For simplicity, use a mask of ones
            masks = torch.ones(
                (inputs.size(0), ACTION_SPACE_SIZE), dtype=torch.float32
            ).to(device)

            # Forward pass
            logits = model(inputs, masks)

            # Compute loss
            loss = loss_fn(logits, targets)
            total_loss += loss.item()
            batch_count += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_loss:.4f}")

        # ------------------------
        # Validation Section
        # ------------------------

        model.eval()  # Set model to evaluation mode
        correct = 0
        total = 0
        val_loss = 0

        with torch.no_grad():
            for inputs, targets in valid_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                masks = torch.ones(
                    (inputs.size(0), ACTION_SPACE_SIZE), dtype=torch.float32
                ).to(device)

                logits = model(inputs, masks)
                val_loss += loss_fn(logits, targets).item()  # Sum validation loss

                # Calculate accuracy
                _, predicted = torch.max(logits, dim=1)
                total += targets.size(0)
                correct += (predicted == targets).sum().item()

        avg_val_loss = val_loss / len(valid_loader)
        accuracy = correct / total
        print(f"Validation Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2%}")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'.")
