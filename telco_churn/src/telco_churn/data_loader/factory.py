# src/data_loader/factory.py
from .csv_loader import CSVLoader
from .base_loader import DataLoader


class DataLoaderFactory:
    @staticmethod
    def get_loader(file_type: str) -> DataLoader:
        if file_type == "csv":
            return CSVLoader()
        raise ValueError(f"Unsupported file type: {file_type}")
