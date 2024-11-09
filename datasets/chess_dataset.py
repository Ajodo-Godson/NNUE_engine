import torch
from torch.utils.data import IterableDataset
import pandas as pd
import chess
import zipfile
from utils.board_repr import board_to_input_tensor
from utils.move_mapping import move_to_index

import random


class ChessIterableDataset(IterableDataset):
    def __init__(self, zip_path, csv_filename, val_split=0.2, seed=42):
        self.zip_path = zip_path
        self.csv_filename = csv_filename
        self.val_split = val_split
        random.seed(seed)

    def __iter__(self):
        with zipfile.ZipFile(self.zip_path) as z:
            with z.open(self.csv_filename) as f:
                # Read in chunks of 1000 rows, but process each row individually
                for chunk in pd.read_csv(f, chunksize=1000):
                    for _, row in chunk.iterrows():
                        # Decide whether this row is for training or validation
                        split = "val" if random.random() < self.val_split else "train"

                        # Extract FEN and evaluation from the row
                        fen = row.get("FEN")
                        eval_str = row.get("Evaluation")

                        # Convert FEN to input tensor
                        input_tensor = board_to_input_tensor(chess.Board(fen))

                        # Parse evaluation to get target index
                        target_index = self.parse_evaluation(eval_str)

                        # Yield data along with split identifier
                        yield input_tensor, target_index, split

    def parse_evaluation(self, eval_str):
        eval_str = str(eval_str)
        if "#" in eval_str:
            return 10000 if "#+" in eval_str else -10000
        try:
            return float(eval_str)
        except ValueError:
            return 0.0
