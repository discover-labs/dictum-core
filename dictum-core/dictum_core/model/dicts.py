from collections import UserDict

import dictum_core.model.calculations
import dictum_core.model.scalar


class DictumObjectDict(UserDict):
    object_name: str

    def __getitem__(self, key: str):
        result = self.data.get(key)
        if result is None:
            raise KeyError(f"{self.object_name} {key} does not exist")
        return result

    def __setitem__(self, name: str, value):
        if name in self.data:
            raise KeyError(f"Duplicate {self.object_name}: {name}")
        return super().__setitem__(name, value)

    def get(self, key: str):
        return self[key]

    def add(self, calc: "dictum_core.model.calculations.Calculation"):
        self[calc.id] = calc


class MeasureDict(DictumObjectDict):
    object_name: str = "measure"

    def get(self, key: str) -> "dictum_core.model.calculations.Measure":
        return super().get(key)


class DimensionDict(DictumObjectDict):
    object_name: str = "dimension"

    def get(self, key: str) -> "dictum_core.model.calculations.Dimension":
        return super().get(key)


class MetricDict(DictumObjectDict):
    object_name: str = "metric"

    def get(self, key: str) -> "dictum_core.model.calculations.Metric":
        return super().get(key)


class TransformDict(DictumObjectDict):
    object_name: str = "scalar transform"

    def get(self, key: str) -> "dictum_core.model.scalar.ScalarTransformMeta":
        return super().get(key)
