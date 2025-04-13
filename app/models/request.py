from pydantic import BaseModel
from typing import List


class CalculationRequest(BaseModel):
    price: List[float]
    support_cost: List[float]
    daily_orders: List[List[float]]
    n_days: int
