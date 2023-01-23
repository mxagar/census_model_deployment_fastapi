"""Additional Pydantic Models used in the census salary API.
Other models/classes such as the data inputs
are defined in the core.py module
of the census salary library.

Author: Mikel Sagardia
Date: 2023-01-23
"""

from typing import List, Optional

from pydantic import BaseModel
# Recall that the other schemas are defined in core.py:
# from census_salary.core.core import DataRow, MultipleDataRows

class Health(BaseModel):
    name: str
    api_version: str
    model_lib_version: str

class PredictionResults(BaseModel):
    model_lib_version: str
    timestamp: str
    predictions: Optional[List[str]]
