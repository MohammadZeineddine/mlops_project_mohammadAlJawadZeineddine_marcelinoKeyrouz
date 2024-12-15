# src/data_loader/csv_loader.py
import pandas as pd

from .base_loader import DataLoader


class CSVLoader(DataLoader):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from a CSV file."""
        return pd.read_csv(file_path)
