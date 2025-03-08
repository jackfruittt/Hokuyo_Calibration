"""
Hokuyo UTM-30LX Calibration Model (Degree 2)
Generated: 2025-03-08 16:17:11.816789
Range: 0.100m to 0.115m
RMSE: 0.31 mm
"""

import numpy as np

# Polynomial coefficients (highest degree first)
coefficients = [
    2.0099508099e+01,
    -4.2696039585e+00,
    3.2411497389e-01
]

def correct_measurement(raw_measurement):
    """
    Apply calibration correction to a raw LIDAR measurement.

    Args:
        raw_measurement: Raw distance measurement in meters

    Returns:
        float: Calibrated distance measurement in meters
    """
    # Using numpy's polyval for stable polynomial evaluation
    return np.polyval(coefficients, raw_measurement)

def correct_measurements(raw_measurements):
    """
    Apply calibration correction to multiple raw measurements.

    Args:
        raw_measurements: List or array of raw distance measurements in meters

    Returns:
        numpy.ndarray: Calibrated distance measurements in meters
    """
    return np.polyval(coefficients, raw_measurements)
