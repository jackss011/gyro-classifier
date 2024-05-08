import numpy as np
from abc import ABC, abstractmethod
import math


class DeltaRegime(ABC):
    name: str

    @abstractmethod
    def get(self, epoch: int) -> float:
        pass


class DeltaRegime_Zero(DeltaRegime):
    name = "zero"

    def get(self, epoch):
        return 0
    

class DeltaRegime_Const(DeltaRegime):
    name = "const"

    def __init__(self, const: float):
        self.const = const

    def get(self, epoch):
        return self.const
    

class DeltaRegime_Linear(DeltaRegime):
    name = "linear"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = min(epoch / self.max_at_epoch, 1)
        return self.min + (self.max - self.min) * perc
    

class DeltaRegime_Exponential(DeltaRegime):
    name = "exp"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

        self.k = math.log(1 + self.max - self.min) / max_at_epoch

    def get(self, epoch):
        return min(self.min + math.exp(self.k * epoch) - 1, self.max)
    

class DeltaRegime_Sqrt(DeltaRegime):
    name = "sqrt"

    def __init__(self, min: float, max: float, max_at_epoch: int):
        self.min = min
        self.max = max
        self.max_at_epoch = max_at_epoch

    def get(self, epoch):
        perc = min(math.sqrt(float(epoch) / self.max_at_epoch), 1)
        return self.min + (self.max - self.min) * perc
    
