from pydantic import BaseModel
from typing import List


class DigitData(BaseModel):
    """
    Represents a single 8x8 handwritten digit image as 64 pixel intensity features.
    Each pixel value ranges from 0.0 to 16.0.
    """
    pixels: List[float]  # expects a list of 64 float values


class DigitBatch(BaseModel):
    """
    Represents a batch of digit samples for batch prediction.
    """
    samples: List[DigitData]


class DigitResponse(BaseModel):
    prediction: int


class BatchResponse(BaseModel):
    predictions: List[int]
