import pandas as pd


from carla import log
from carla.data.catalog import DataCatalog
from sklearn.model_selection import train_test_split
from typing import List


class DynamicCsvCatalog(DataCatalog):
    """
    Wrapper class for the DataCatalog similar to the built-in CsvCatalog
    but with new capabilities required to control data in the experiments.
    Attributes:
        file_path (str):
            Path to the .csv file containing the dataset.
        categorical (List[str]):
            Names of columns describing categorical features.
        continuous (List[str]):
            Names of columns describing continuous (i.e. numerical) features.
        immutables (List[str]):
            Names of columns describing immutable features, not supported by all generators.
        target (str):
            Name of the column that contains the target variable.
        test_size (float):
            Proportion of the dataset which should be withheld as an independent test set.
    """
    def __init__(self, file_path: str, categorical: List[str],  continuous: List[str],
                 immutables: List[str], target: str, test_size: float = 0.5,
                 scaling_method: str = "MinMax", encoding_method: str = "OneHot_drop_binary",
                 positive=1, negative=0):

        self._categorical = categorical
        self._continuous = continuous
        self._immutables = immutables
        self._target = target
        self._positive = positive
        self._negative = negative

        # Load the raw data
        raw = pd.read_csv(file_path)
        train_raw, test_raw = train_test_split(raw, test_size=test_size, stratify=raw[target])
        log.info(f"Balance: train set {train_raw[self.target].mean()}, test set {test_raw[self.target].mean()}")
        super().__init__("custom", raw, train_raw, test_raw,
                         scaling_method, encoding_method)

    @property
    def categorical(self) -> List[str]:
        return self._categorical

    @property
    def continuous(self) -> List[str]:
        return self._continuous

    @property
    def immutables(self) -> List[str]:
        return self._immutables

    @property
    def target(self) -> str:
        return self._target

    @property
    def positive(self) -> str:
        return self._positive

    @property
    def negative(self) -> str:
        return self._negative
