from .base_loader import DataLoader
from .csv_loader import CSVLoader


class DataLoaderFactory:
    """Factory class for creating data loaders."""

    @staticmethod
    def get_loader(file_type: str) -> DataLoader:
        """Gets a data loader instance based on the file type.

        Args:
            file_type (str): The type of the file (e.g., "csv").

        Returns:
            DataLoader: An instance of the appropriate DataLoader subclass.

        Raises:
            ValueError: If the specified file type is not supported.
        """
        if file_type == "csv":
            return CSVLoader()
        raise ValueError(f"Unsupported file type: {file_type}")
