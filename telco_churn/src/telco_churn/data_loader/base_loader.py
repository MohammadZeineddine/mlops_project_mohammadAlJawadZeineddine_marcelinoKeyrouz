from abc import ABC, abstractmethod

import pandas as pd


class DataLoader(ABC):
    """Abstract base class for data loaders.

    This class defines the interface for loading data from different sources.
    """

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from the given file path.

        Args:
            file_path (str): The path to the data file.

        Returns:
            pd.DataFrame: The loaded data as a pandas DataFrame.

        Raises:
            NotImplementedError: This method is abstract and must be implemented
                by subclasses.
        """
        pass