"""
Hokuyo UTM-30LX Calibration Model (Degree 3)
Generated: 2025-03-08 16:17:12.251877
Range: 0.100m to 0.115m
RMSE: 0.00 mm
"""

import numpy as np

# Polynomial coefficients (highest degree first)
coefficients = [
    2.3512307696e+03,
    -8.7408584403e+02,
    1.0891634616e+02,
    -4.4443624943e+00
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
