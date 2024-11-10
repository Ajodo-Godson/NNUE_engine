import torch
import torch.utils.data as data
import pandas as pd
import chess
import zipfile
import logging
import hashlib
from utils.board_repr import board_to_input_tensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessIterableDataset(data.IterableDataset):
    def __init__(self, zip_path, csv_filename, val_split=0.2, seed=42):
        self.zip_path = zip_path
        self.csv_filename = csv_filename
        self.val_split = val_split
        self.seed = seed

    def parse_evaluation(self, eval_str):
        eval_str = str(eval_str).strip()

        if eval_str.startswith("#"):
            # Handle mate scores
            if eval_str.startswith("#-"):
                # Mate in Black's favor
                eval_score = -10000.0
            elif eval_str.startswith("#+"):
                # Mate in White's favor
                eval_score = 10000.0
            else:
                # Handle other possible formats, default to 0.0 or assign a default high value
                eval_score = 10000.0  # Assuming mate in White's favor
        else:
            try:
                # Convert centipawn evaluations to float
                eval_score = float(eval_str)
            except ValueError:
                # Default to 0.0 if parsing fails
                eval_score = 0.0

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
        try:
            halfmove_clock = int(parts[4])
        except (IndexError, ValueError):
            halfmove_clock = 0
        metadata.append(halfmove_clock / 100.0)

        # Fullmove number (optional)
        try:
            fullmove_number = int(parts[5])
        except (IndexError, ValueError):
            fullmove_number = 1
        metadata.append(fullmove_number / 100.0)

        # Convert to PyTorch tensor
        metadata_tensor = torch.tensor(metadata, dtype=torch.float32)
        return metadata_tensor

    def get_deterministic_hash(self, s):
        return int(hashlib.md5(s.encode("utf-8")).hexdigest(), 16)

    def __iter__(self):
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(self.csv_filename) as f:
                for chunk in pd.read_csv(
                    f,
                    chunksize=1000,
                    dtype={"FEN": str, "Evaluation": str},
                    encoding="utf-8",
                ):
                    for idx, row in chunk.iterrows():
                        fen = row.get("FEN")
                        eval_str = row.get("Evaluation")

                        # Handle NaN values
                        if pd.isna(fen) or pd.isna(eval_str):
                            logger.warning(
                                f"NaN FEN or Evaluation at row {idx}. Skipping."
                            )
                            continue

                        # Clean strings
                        fen = str(fen).strip().replace("\u00A0", " ")
                        eval_str = str(eval_str).strip().replace("\u00A0", " ")

                        # Check for empty strings
                        if not fen or not eval_str:
                            logger.warning(
                                f"Empty FEN or Evaluation at row {idx}. FEN: '{fen}', Evaluation: '{eval_str}'. Skipping."
                            )
                            continue

                        # Use deterministic hash for split
                        hash_value = self.get_deterministic_hash(fen)
                        split = (
                            "val"
                            if (hash_value % 100) < self.val_split * 100
                            else "train"
                        )

                        # Convert FEN to input tensor and metadata tensor
                        try:
                            board = chess.Board(fen)
                            input_tensor = board_to_input_tensor(board)
                            metadata_tensor = self.extract_metadata(fen)
                        except ValueError as e:
                            logger.warning(
                                f"Invalid FEN at row {idx}: {fen}. Error: {e}. Skipping."
                            )
                            continue

                        # Parse evaluation to get target value
                        target_value = self.parse_evaluation(eval_str)

                        # Yield data along with split identifier
                        yield input_tensor, metadata_tensor, target_value, split
