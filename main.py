import torch
import torch.nn as nn
import torch.optim as optim
import chess
import chess.engine
import random
from torch.utils.data import DataLoader
import os

# Import custom modules
from utils.board_repr import board_to_input_tensor
from utils.move_mapping import move_to_index
from utils.constants import INPUT_SIZE, HIDDEN_SIZE, ACTION_SPACE_SIZE
from models.neural import ChessNet
from datasets.chess_dataset import ChessDataset

# Set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths
STOCKFISH_PATH = "/opt/homebrew/bin/stockfish"
DATASET_PATH = "data/chess_dataset.pt"
MODEL_SAVE_PATH = "saved_models/chess_model.pth"

# Parameters
NUM_POSITIONS = 10000
BATCH_SIZE = 32
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

# Create directories if they don't exist
os.makedirs("data", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

# ------------------------
# Data Generation Section
# ------------------------


def generate_dataset(num_positions, stockfish_path, save_path):
    engine = chess.engine.SimpleEngine.popen_uci(stockfish_path)
    dataset = []

    for i in range(num_positions):
        board = chess.Board()
        num_moves = random.randint(0, 40)
        for _ in range(num_moves):
            if board.is_game_over():
                break
            move = random.choice(list(board.legal_moves))
            board.push(move)
        if board.is_game_over():
            continue
        try:
            result = engine.play(board, chess.engine.Limit(depth=10))
            best_move = result.move
            input_tensor = board_to_input_tensor(board)
            target_index = move_to_index(best_move)
            dataset.append((input_tensor, target_index))
        except Exception as e:
            print(f"Error at position {i}: {e}")
            continue
        if (i + 1) % 100 == 0:
            print(f"Generated {i + 1}/{num_positions} positions.")

    engine.quit()
    torch.save(dataset, save_path)
    print(f"Dataset saved to '{save_path}'.")


# Check if dataset exists; if not, generate it
if not os.path.isfile(DATASET_PATH):
    print("Generating dataset...")
    generate_dataset(NUM_POSITIONS, STOCKFISH_PATH, DATASET_PATH)
else:
    print(f"Dataset found at '{DATASET_PATH}'. Skipping data generation.")

# ------------------------
# Data Loading Section
# ------------------------

# Load the dataset
dataset = torch.load(DATASET_PATH)
print(f"Loaded dataset with {len(dataset)} positions.")

# Split the dataset
train_size = int(0.8 * len(dataset))
valid_size = len(dataset) - train_size
train_data, valid_data = torch.utils.data.random_split(
    dataset, [train_size, valid_size]
)

# Create DataLoaders
train_dataset = ChessDataset(train_data)
valid_dataset = ChessDataset(valid_data)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE)

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
    for inputs, targets in train_loader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        # For simplicity, use a mask of ones
        masks = torch.ones((inputs.size(0), ACTION_SPACE_SIZE), dtype=torch.float32).to(
            device
        )

        # Forward pass
        logits = model(inputs, masks)

        # Compute loss
        loss = loss_fn(logits, targets)
        total_loss += loss.item()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    avg_loss = total_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Loss: {avg_loss:.4f}")

    # Validation
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in valid_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            masks = torch.ones(
                (inputs.size(0), ACTION_SPACE_SIZE), dtype=torch.float32
            ).to(device)

            logits = model(inputs, masks)
            _, predicted = torch.max(logits, dim=1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    accuracy = correct / total
    print(f"Validation Accuracy: {accuracy:.2%}")

# Save the model
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"Model saved to '{MODEL_SAVE_PATH}'.")
