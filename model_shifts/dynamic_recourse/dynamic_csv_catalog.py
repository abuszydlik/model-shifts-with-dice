import pandas as pd


from carla import log
from carla.data.catalog import DataCatalog
from sklearn.base import BaseEstimator
from sklearn.model_selection import train_test_split
from carla.data.pipelining import (decode,
                                   descale,
                                   encode,
                                   fit_encoder,
                                   fit_scaler,
                                   scale)
from typing import Callable, List, Tuple


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
        self.name = "custom"

        # Load the raw data
        raw = pd.read_csv(file_path)
        train_raw, test_raw = train_test_split(raw, test_size=test_size, stratify=raw[target])
        log.info(f"Balance: train set {train_raw[self.target].mean()}, test set {test_raw[self.target].mean()}")

        # Fit scaler and encoder
        if len(self.continuous) == 0:
            self.scaler: BaseEstimator = None
        else:
            self.scaler: BaseEstimator = fit_scaler(scaling_method, raw[self.continuous])

        self.encoder: BaseEstimator = fit_encoder(encoding_method, raw[self.categorical])

        self._identity_encoding = (encoding_method is None or encoding_method == "Identity")

        # Preparing pipeline components
        self._pipeline = self.__init_pipeline()
        self._inverse_pipeline = self.__init_inverse_pipeline()

        # Process the data
        self._df = self.transform(raw)
        self._df_train = self.transform(train_raw)
        self._df_test = self.transform(test_raw)

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

    def __init_pipeline(self) -> List[Tuple[str, Callable]]:
        result = []
        if self.scaler is not None:
            result.append(("scaler", lambda x: scale(self.scaler, self.continuous, x)))
        if self.encoder is not None:
            result.append(("encoder", lambda x: encode(self.encoder, self.categorical, x)))
        return result

    def __init_inverse_pipeline(self) -> List[Tuple[str, Callable]]:
        result = []
        if self.encoder is not None:
            result.append(("encoder", lambda x: decode(self.encoder, self.categorical, x)))
        if self.scaler is not None:
            result.append(("scaler", lambda x: descale(self.scaler, self.continuous, x)))
        return result
