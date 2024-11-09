import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os

# Import custom modules
from utils.board_repr import board_to_input_tensor
from utils.constants import INPUT_SIZE, HIDDEN_SIZE
from models.neural import ChessNet
from datasets.chess_dataset import ChessIterableDataset

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths and parameters
ZIP_PATH = "data/datasets.zip"
CSV_FILENAME = "datasets/tactic_evals.csv"
MODEL_SAVE_PATH = "saved_models/chess_model.pth"
PROCESSED_DATA_PATH = "processed_data_regression.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.0001  # Reduced learning rate
METADATA_SIZE = 7  # Number of metadata features

# Create directories if they don't exist
os.makedirs("saved_models", exist_ok=True)

if __name__ == "__main__":
    # ------------------------
    # Data Loading Section
    # ------------------------

    print("Loading dataset...")

    if os.path.exists(PROCESSED_DATA_PATH):
        print("Processed dataset found. Loading...")
        data = torch.load(PROCESSED_DATA_PATH, weights_only=True)
        train_data_list = data["train"]
        val_data_list = data["val"]
    else:
        print("Processed dataset not found. Generating...")
        dataset = ChessIterableDataset(ZIP_PATH, CSV_FILENAME, val_split=0.2)
        train_data_list = []
        val_data_list = []

        for i, (input_tensor, metadata_tensor, target_value, split) in enumerate(
            dataset
        ):
            if split == "train":
                train_data_list.append((input_tensor, metadata_tensor, target_value))
            else:
                val_data_list.append((input_tensor, metadata_tensor, target_value))
            if i % 1000 == 0:
                print(f"Processed {i} samples from the dataset...")

        # Save the processed data
        torch.save(
            {"train": train_data_list, "val": val_data_list}, PROCESSED_DATA_PATH
        )
        print("Processed dataset saved.")

    # Create DataLoaders
    train_loader = DataLoader(train_data_list, batch_size=BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(val_data_list, batch_size=BATCH_SIZE, shuffle=False)

    # After loading the processed data
    print(f"First training sample input_tensor shape: {train_data_list[0][0].shape}")
    print(f"First training sample metadata_tensor shape: {train_data_list[0][1].shape}")
    print(f"First training sample target_value: {train_data_list[0][2]}")

# In the training loop
for batch_idx, (inputs, metadata, targets) in enumerate(train_loader):
    print(
        f"Batch {batch_idx}, inputs shape: {inputs.shape}, metadata shape: {metadata.shape}, targets shape: {targets.shape}"
    )
    # Proceed with the rest of your training loop...

    # ------------------------
    # Model Training Section
    # ------------------------

    # Initialize model, loss function, and optimizer
    model = ChessNet(metadata_size=METADATA_SIZE).to(device)
    loss_fn = nn.MSELoss()  # Or nn.L1Loss() for MAE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Training loop
    print("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        batch_count = 0
        for batch_idx, (inputs, metadata, targets) in enumerate(train_loader):
            inputs = inputs.to(device)  # Shape: [batch_size, 1, 8, 8]
            metadata = metadata.to(device)  # Shape: [batch_size, METADATA_SIZE]
            targets = targets.to(device).float()

            # Forward pass
            outputs = model(inputs, metadata)

            # Compute loss
            loss = loss_fn(outputs, targets)
            total_loss += loss.item()
            batch_count += 1

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Logging
            if batch_idx % 100 == 0:
                print(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx+1}/{len(train_loader)}], "
                    f"Loss: {loss.item():.6f}"
                )

        avg_loss = total_loss / batch_count
        print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Training Loss: {avg_loss:.6f}")

        # ------------------------
        # Validation Section
        # ------------------------

        model.eval()
        val_loss = 0
        total_samples = 0
        total_mae = 0

        with torch.no_grad():
            for batch_idx, (inputs, metadata, targets) in enumerate(valid_loader):
                inputs = inputs.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device).float()
                outputs = model(inputs, metadata)

                loss = loss_fn(outputs, targets)
                val_loss += loss.item()
                total_samples += targets.size(0)
                total_mae += torch.sum(torch.abs(outputs - targets)).item()

                if batch_idx % 100 == 0:
                    print(
                        f"Validation Batch [{batch_idx+1}/{len(valid_loader)}], "
                        f"Batch Loss: {loss.item():.6f}"
                    )

        avg_val_loss = val_loss / len(valid_loader)
        avg_mae = total_mae / total_samples
        print(f"Validation Loss: {avg_val_loss:.6f}, MAE: {avg_mae:.6f}")

    # Save the model
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    print(f"Model saved to '{MODEL_SAVE_PATH}'.")
