"""
Hokuyo UTM-30LX Calibration Model (Degree 1)
Generated: 2025-03-08 16:17:11.104746
Range: 0.100m to 0.115m
RMSE: 0.75 mm
"""

import numpy as np

# Polynomial coefficients (highest degree first)
coefficients = [
    8.2031606833e-01,
    2.8139977467e-03
]

def correct_measurement(raw_measurement):
    """
    Apply calibration correction to a raw LIDAR measurement.

    Args:
        raw_measurement: Raw distance measurement in meters

    Returns:
        float: Calibrated distance measurement in meters
    """
    return coefficients[0] * raw_measurement + coefficients[1]

def correct_measurements(raw_measurements):
    """
    Apply calibration correction to multiple raw measurements.

    Args:
        raw_measurements: List or array of raw distance measurements in meters

    Returns:
        numpy.ndarray: Calibrated distance measurements in meters
    """
    return np.polyval(coefficients, raw_measurements)
