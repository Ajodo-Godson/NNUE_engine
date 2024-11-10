import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import logging

# Import custom modules
from utils.board_repr import board_to_input_tensor
from models.neural import ChessNet
from datasets.chess_dataset import ChessIterableDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, =
    format="%(asctime)s %(levelname)s:%(message)s",
    handlers=[logging.FileHandler("training.log"), logging.StreamHandler()],
)
logger = logging.getLogger(__name__)

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
logger.info(f"Using device: {device}")

# Paths and parameters
ZIP_PATH = "data/datasets.zip"
CSV_FILENAME = "datasets/chessData.csv"
H5_PATH = "data/chess_data.h5"
MODEL_SAVE_PATH = "saved_models/chess_model.pth"
BATCH_SIZE = 32
NUM_EPOCHS = 3  # Reduced for testing
LEARNING_RATE = 0.0001  # Reduced learning rate
METADATA_SIZE = 8  # Updated from 7 to 8
NUM_WORKERS = 4  

# Create directories if they don't exist
os.makedirs("saved_models", exist_ok=True)

if __name__ == "__main__":
    # ------------------------
    # Data Preprocessing Section
    # ------------------------

    # Check if HDF5 file exists
    if not os.path.exists(H5_PATH):
        logger.info("HDF5 file not found. Creating HDF5 file...")
        dataset = ChessIterableDataset(
            zip_path=ZIP_PATH,
            csv_filename=CSV_FILENAME,
            val_split=0.2,
            split="train",  # Split handled within save_to_hdf5
            h5_path=H5_PATH,
            metadata_size=METADATA_SIZE,  # Pass updated metadata_size
        )
        dataset.save_to_hdf5(H5_PATH)  # Remove max_samples
        logger.info("HDF5 file created successfully.")

    # ------------------------
    # Data Loading Section
    # ------------------------

    logger.info("Initializing datasets...")

    # Create dataset instances for training and validation from HDF5
    train_dataset = ChessIterableDataset(
        zip_path=ZIP_PATH,
        csv_filename=CSV_FILENAME,
        val_split=0.2,
        split="train",
        h5_path=H5_PATH,
        metadata_size=METADATA_SIZE,  # Pass updated metadata_size
    )

    val_dataset = ChessIterableDataset(
        zip_path=ZIP_PATH,
        csv_filename=CSV_FILENAME,
        val_split=0.2,
        split="val",
        h5_path=H5_PATH,
        metadata_size=METADATA_SIZE,  # Pass updated metadata_size
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,  # Shuffle not needed for IterableDataset
        num_workers=NUM_WORKERS,  # Set to 4
        pin_memory=True,
    )

    valid_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,  # Set to 4
        pin_memory=True,
    )

    # ------------------------
    # Model Training Section
    # ------------------------

    # Initialize model, loss function, and optimizer
    model = ChessNet(metadata_size=METADATA_SIZE).to(device)
    loss_fn = nn.MSELoss()  # Or nn.L1Loss() for MAE
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

    # Initialize GradScaler only if CUDA is available
    use_amp = device.type == "cuda"
    if use_amp:
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Using Automatic Mixed Precision (AMP).")
    else:
        scaler = None
        logger.info("AMP not enabled. Training on CPU.")

    # Initialize best validation loss
    best_val_loss = float("inf")

    logger.info("Starting training...")
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0.0
        batch_count = 0

        # Reset batch_idx for each epoch by starting enumerate at 1
        for batch_idx, (inputs, metadata, targets) in enumerate(train_loader, 1):
            inputs = inputs.to(device)  # Shape: [batch_size, 1, 8, 8]
            metadata = metadata.to(device)  # Shape: [batch_size, 8]
            targets = targets.to(device).float()  # Shape: [batch_size]

            optimizer.zero_grad()

            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(inputs, metadata).squeeze()  # Shape: [batch_size]
                    loss = loss_fn(outputs, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(inputs, metadata).squeeze()  # Shape: [batch_size]
                loss = loss_fn(outputs, targets)
                loss.backward()
                optimizer.step()

            total_loss += loss.item()
            batch_count += 1

            # Logging
            if batch_idx % 100 == 0:
                logger.info(
                    f"Epoch [{epoch+1}/{NUM_EPOCHS}], Batch [{batch_idx}], Loss: {loss.item():.6f}"
                )

        # Adjust learning rate
        scheduler.step()

        avg_loss = total_loss / batch_count
        logger.info(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Average Training Loss: {avg_loss:.6f}"
        )

        # ------------------------
        # Validation Section
        # ------------------------

        model.eval()
        val_loss = 0.0
        total_samples = 0
        total_mae = 0.0

        with torch.no_grad():
            for batch_idx, (inputs, metadata, targets) in enumerate(valid_loader, 1):
                inputs = inputs.to(device)
                metadata = metadata.to(device)
                targets = targets.to(device).float()

                if use_amp:
                    with torch.cuda.amp.autocast():
                        outputs = model(inputs, metadata).squeeze()
                        loss = loss_fn(outputs, targets)
                else:
                    outputs = model(inputs, metadata).squeeze()
                    loss = loss_fn(outputs, targets)

                val_loss += loss.item()

                # Calculate MAE
                mae = torch.abs(outputs - targets).sum().item()
                total_mae += mae

                total_samples += targets.size(0)

                if batch_idx % 100 == 0:
                    logger.info(
                        f"Validation Batch [{batch_idx}], Loss: {loss.item():.6f}"
                    )

        avg_val_loss = val_loss / batch_idx
        avg_mae = total_mae / total_samples
        logger.info(
            f"Epoch [{epoch+1}/{NUM_EPOCHS}], Validation Loss: {avg_val_loss:.6f}, MAE: {avg_mae:.6f}"
        )

        # Save the best model based on validation loss
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            logger.info(f"Best model saved with Validation Loss: {best_val_loss:.6f}")

    # Save the final model if not already saved
    torch.save(model.state_dict(), MODEL_SAVE_PATH)
    logger.info(f"Final model saved to '{MODEL_SAVE_PATH}'.")
