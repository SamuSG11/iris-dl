import pandas as pd
from typing import List, Optional

from sklearn.preprocessing import StandardScaler

from lib.encoder import GenericEncoder
from src.utils import get_logger, PreprocessingError

logger = get_logger(__name__)


class FeatureScaler:

    def __init__(self):
        self.scaler = None
        self.columns = None
        self.scaler_type = None

    def fit(self, df: pd.DataFrame, columns: Optional[List[str]] = None):

        if columns is None:
            columns = df.select_dtypes(include=['number']).columns.tolist()

        self.columns = columns

        self.scaler = StandardScaler()
        self.scaler.fit(df[columns])

        self.scaler_type = "standard"

        return self

    def transform(self, df: pd.DataFrame):

        if self.scaler is None:
            raise Exception("Scaler not fitted")

        result = df.copy()
        result[self.columns] = self.scaler.transform(df[self.columns])

        return result

    def fit_transform(self, df: pd.DataFrame, columns: Optional[List[str]] = None):
        self.fit(df, columns)
        return self.transform(df)
