from carla.data.catalog import OnlineCatalog


class DynamicOnlineCatalog(OnlineCatalog):

    def __init__(self, data_name, target, positive=1, negative=0,
                 scaling_method: str = "MinMax", encoding_method: str = "OneHot_drop_binary"):

        self._target = target
        self._positive = positive
        self._negative = negative
        super().__init__(data_name, scaling_method, encoding_method)

    @property
    def positive(self) -> str:
        return self._positive

    @property
    def negative(self) -> str:
        return self._negative
