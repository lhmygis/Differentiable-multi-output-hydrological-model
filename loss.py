"""
This file is part of the accompanying code to our manuscript:

Jiang S., Zheng Y., & Solomatine D.. (2020) Improving AI system awareness of geoscience knowledge: Symbiotic integration of physical approaches and deep learning. Geophysical Research Letters, 47, e2020GL088229. https://doi.org/10.1029/2020GL088229

Copyright (c) 2020 Shijie Jiang. All rights reserved.

You should have received a copy of the Apache-2.0 license along with the code. If not,
see <https://opensource.org/licenses/Apache-2.0>
"""

import keras.backend as K
import numpy as np
from scipy import stats
import logging
import tensorflow as tf


def nse_loss(y_true, y_pred):

    #y_pred = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]

    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)

    return numerator / (denominator+0.5)


def nse_metrics(y_true, y_pred):

    #y_pred = K.permute_dimensions(y_pred, pattern=(1,0,2))  #[2212,60,1] ->  [60,2218,1]


    y_true = y_true[:, :, :]  # Omit values in the spinup period (the first 365 days)
    y_pred = y_pred[:, :, :]  # Omit values in the spinup period (the first 365 days)

    numerator = K.sum(K.square(y_pred - y_true), axis=1)
    denominator = K.sum(K.square(y_true - K.mean(y_true, axis=1, keepdims=True)), axis=1)
    rNSE = numerator / (denominator+0.5)

    return 1.0 - rNSE



