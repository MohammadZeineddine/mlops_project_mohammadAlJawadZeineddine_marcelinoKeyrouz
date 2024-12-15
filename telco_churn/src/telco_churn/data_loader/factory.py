# src/data_loader/factory.py
from .base_loader import DataLoader
from .csv_loader import CSVLoader


class DataLoaderFactory:
    @staticmethod
    def get_loader(file_type: str) -> DataLoader:
        if file_type == "csv":
            return CSVLoader()
        raise ValueError(f"Unsupported file type: {file_type}")
