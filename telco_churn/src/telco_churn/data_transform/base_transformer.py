from abc import ABC, abstractmethod

import pandas as pd


class BaseTransformer(ABC):
    @abstractmethod
    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """Apply transformation to the data."""
        pass
