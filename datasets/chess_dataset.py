import torch
import torch.utils.data as data
import pandas as pd
import chess
import zipfile
from utils.board_repr import board_to_input_tensor


class ChessIterableDataset(data.IterableDataset):
    def __init__(self, zip_path, csv_filename, val_split=0.2, seed=42):
        self.zip_path = zip_path
        self.csv_filename = csv_filename
        self.val_split = val_split
        self.seed = seed

    def parse_evaluation(self, eval_str):
        eval_str = str(eval_str)
        if "#" in eval_str:
            # Handle mate scores
            if eval_str.startswith("#-"):
                eval_score = -10000.0  # Mate in Black's favor
            elif eval_str.startswith("#+"):
                eval_score = 10000.0  # Mate in White's favor
            else:
                eval_score = 10000.0  # Assume mate in White's favor
        else:
            try:
                eval_score = float(eval_str)
            except ValueError:
                eval_score = 0.0  # Default to 0 if parsing fails

        # Normalize the evaluation score
        eval_score = max(
            min(eval_score, 1000.0), -1000.0
        )  # Clamp between -1000 and 1000
        eval_score /= 1000.0  # Scale to range [-1.0, 1.0]

        return eval_score

    def extract_metadata(self, fen):
        parts = fen.split(" ")
        metadata = []
        # Player to move
        metadata.append(1.0 if parts[1] == "w" else 0.0)
        # Castling rights
        castling_rights = parts[2]
        metadata.extend(
            [
                1.0 if "K" in castling_rights else 0.0,
                1.0 if "Q" in castling_rights else 0.0,
                1.0 if "k" in castling_rights else 0.0,
                1.0 if "q" in castling_rights else 0.0,
            ]
        )
        # En passant target square
        metadata.append(0.0 if parts[3] == "-" else 1.0)
        # Halfmove clock (normalized)
        halfmove_clock = int(parts[4])
        metadata.append(halfmove_clock / 100.0)
        # Convert to PyTorch tensor
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32)
        return metadata_tensor

    def __iter__(self):
        import random

        random.seed(self.seed)
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(self.csv_filename) as f:
                for chunk in pd.read_csv(f, chunksize=1000):
                    for _, row in chunk.iterrows():
                        # Randomly assign to train or validation set
                        split = "val" if random.random() < self.val_split else "train"

                        # Extract FEN and evaluation
                        fen = row.get("FEN")
                        eval_str = row.get("Evaluation")

                        # Convert FEN to input tensor and metadata tensor
                        board = chess.Board(fen)
                        input_tensor = board_to_input_tensor(board)
                        metadata_tensor = self.extract_metadata(fen)

                        # Parse evaluation to get target value
                        target_value = self.parse_evaluation(eval_str)

                        # Yield data along with split identifier
                        yield input_tensor, metadata_tensor, target_value, split
