"""
Problem definitions for data generation.

This package contains concrete implementations of Problem1D and ProblemND base classes
for different example problems.
"""

from .example01 import Example01Problem
from .example02 import Example02Problem
from .example03 import Example03Problem
from .example04 import Example04Problem
from .example05 import Example05Problem
from .example06 import Example06Problem

from .sine_product_nd import SineProductProblem

__all__ = [
    'Example01Problem',      # 2D
    'Example02Problem',      # 2D
    'Example03Problem',      # 2D
    'Example04Problem',      # 2D
    'Example05Problem',      # 2D
    'Example06Problem',      # 1D
    'SineProductProblem',    # nD
]
