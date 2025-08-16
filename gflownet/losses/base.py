"""
Base class for GFlowNet losses or objective functions.

.. warning::

    Should not be used directly, but subclassed to implement specific losses or
    objective functions for training a GFlowNet.
"""

import os
from abc import ABCMeta, abstractmethod


class BaseLoss(metaclass=ABCMeta):
    def __init__(self):
        """
        Base class for GFlowNet losses.
        """
