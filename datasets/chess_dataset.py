import torch
import torch.utils.data as data
import pandas as pd
import chess
import zipfile
import logging
import hashlib
import h5py
from utils.board_repr import board_to_input_tensor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChessIterableDataset(data.IterableDataset):
    def __init__(
        self,
        zip_path,
        csv_filename,
        val_split=0.2,
        seed=42,
        split="train",
        h5_path=None,
        metadata_size=8,  # Updated to 8
    ):
        super(ChessIterableDataset, self).__init__()
        assert split in ["train", "val"], "split must be 'train' or 'val'"
        self.zip_path = zip_path
        self.csv_filename = csv_filename
        self.val_split = val_split
        self.seed = seed
        self.split = split
        self.h5_path = h5_path
        self.metadata_size = metadata_size  # Initialize metadata_size

    def parse_evaluation(self, eval_str):
        eval_str = str(eval_str).strip()

        if eval_str.startswith("#"):
            # Handle mate scores
            if eval_str.startswith("#-"):
                # Mate in Black's favor
                eval_score = -1.0
            elif eval_str.startswith("#+"):
                # Mate in White's favor
                eval_score = 1.0
            else:
                # Handle other possible formats, default to 1.0
                eval_score = 1.0  # Assuming mate in White's favor
        else:
            try:
                # Convert centipawn evaluations to float
                eval_score = float(eval_str)
            except ValueError:
                # Default to 0.0 if parsing fails
                eval_score = 0.0

        # Normalize centipawn scores to range [-1.0, 1.0]
        if eval_score != 1.0 and eval_score != -1.0:
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

        # Fullmove number
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
        if self.h5_path:
            # Open the HDF5 file within __iter__ so each worker gets its own file handle
            with h5py.File(self.h5_path, "r") as h5_file:
                if self.split not in h5_file:
                    raise KeyError(
                        f"Group '{self.split}' does not exist in the HDF5 file."
                    )
                dataset = h5_file[self.split]
                length = dataset["inputs"].shape[0]

                for i in range(length):
                    input_tensor = torch.tensor(
                        dataset["inputs"][i], dtype=torch.float32
                    )
                    metadata_tensor = torch.tensor(
                        dataset["metadata"][i], dtype=torch.float32
                    )
                    target_value = torch.tensor(
                        dataset["targets"][i], dtype=torch.float32
                    )
                    yield input_tensor, metadata_tensor, target_value
        else:
            # Streaming from CSV
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

                            # Only yield samples matching the specified split
                            if split != self.split:
                                continue

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

                            # Yield data
                            yield input_tensor, metadata_tensor, target_value

    def save_to_hdf5(self, h5_path, max_samples=None):
        """
        Save the dataset to an HDF5 file, split into 'train' and 'val'.

        Args:
            h5_path (str): Path to the HDF5 file to create.
            max_samples (int, optional): Maximum number of samples to process.
        """
        with h5py.File(h5_path, "w") as h5f:
            for split in ["train", "val"]:
                group = h5f.create_group(split)
                group.create_dataset(
                    "inputs",
                    shape=(0, 1, 8, 8),
                    maxshape=(None, 1, 8, 8),
                    dtype="float32",
                    compression="gzip",
                    chunks=(1, 1, 8, 8),
                )
                group.create_dataset(
                    "metadata",
                    shape=(0, self.metadata_size),
                    maxshape=(None, self.metadata_size),
                    dtype="float32",
                    compression="gzip",
                    chunks=(1, self.metadata_size),
                )
                group.create_dataset(
                    "targets",
                    shape=(0,),
                    maxshape=(None,),
                    dtype="float32",
                    compression="gzip",
                    chunks=(1,),
                )

            count = {"train": 0, "val": 0}
            with zipfile.ZipFile(self.zip_path) as z:
                with z.open(self.csv_filename) as f:
                    for chunk in pd.read_csv(
                        f,
                        chunksize=1000,
                        dtype={"FEN": str, "Evaluation": str},
                        encoding="utf-8",
                    ):
                        for idx, row in chunk.iterrows():
                            if max_samples and (
                                count["train"] + count["val"] >= max_samples
                            ):
                                return
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

                            # Append to HDF5
                            group = h5f[split]
                            # Resize datasets
                            group["inputs"].resize((count[split] + 1, 1, 8, 8))
                            group["metadata"].resize(
                                (count[split] + 1, self.metadata_size)
                            )
                            group["targets"].resize((count[split] + 1,))

                            # Assign data
                            group["inputs"][count[split]] = input_tensor.numpy()
                            group["metadata"][count[split]] = metadata_tensor.numpy()
                            group["targets"][count[split]] = target_value

                            count[split] += 1

                            # Logging progress
                            if count["train"] % 100000 == 0 and split == "train":
                                logger.info(f"Saved {count['train']} train samples...")
                            if count["val"] % 100000 == 0 and split == "val":
                                logger.info(f"Saved {count['val']} val samples...")

            logger.info(
                f"HDF5 file '{h5_path}' created successfully with {count['train']} train samples and {count['val']} val samples."
            )
