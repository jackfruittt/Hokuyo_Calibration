#!/usr/bin/env python3
"""
Calibration Module for Hokuyo UTM-30LX LIDAR

This module provides core functionality for LIDAR calibration,
including model generation and application.
"""

import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime

class CalibrationModel:
    """Class representing a LIDAR calibration model."""
    
    def __init__(self, coefficients, degree, start_dist, end_dist, rmse=None):
        """
        Initialize the calibration model.
        
        Args:
            coefficients: Polynomial coefficients (highest degree first)
            degree: Polynomial degree
            start_dist: Calibration range start (meters)
            end_dist: Calibration range end (meters)
            rmse: Root mean square error (mm)
        """
        self.coefficients = coefficients
        self.degree = degree
        self.start_dist = start_dist
        self.end_dist = end_dist
        self.rmse = rmse
    
    def correct_measurement(self, raw_measurement):
        """
        Apply calibration correction to a raw LIDAR measurement.
        
        Args:
            raw_measurement: Raw distance measurement in meters
            
        Returns:
            float: Calibrated distance measurement in meters
        """
        return np.polyval(self.coefficients, raw_measurement)
    
    def correct_measurements(self, raw_measurements):
        """
        Apply calibration correction to multiple raw measurements.
        
        Args:
            raw_measurements: List or array of raw distance measurements in meters
            
        Returns:
            numpy.ndarray: Calibrated distance measurements in meters
        """
        return np.polyval(self.coefficients, raw_measurements)
    
    def save_to_file(self, filename):
        """
        Save the calibration model to a Python file.
        
        Args:
            filename: Output file path
        """
        with open(filename, 'w') as f:
            f.write('"""\n')
            f.write(f"Hokuyo UTM-30LX Calibration Model (Degree {self.degree})\n")
            f.write(f"Generated: {datetime.now()}\n")
            f.write(f"Range: {self.start_dist:.3f}m to {self.end_dist:.3f}m\n")
            if self.rmse is not None:
                f.write(f"RMSE: {self.rmse:.2f} mm\n")
            f.write('"""\n\n')
            
            f.write("import numpy as np\n\n")
            
            # Write coefficients array
            f.write("# Polynomial coefficients (highest degree first)\n")
            f.write("coefficients = [\n")
            for i, c in enumerate(self.coefficients):
                f.write(f"    {c:.10e}")
                if i < len(self.coefficients) - 1:
                    f.write(",")
                f.write("\n")
            f.write("]\n\n")
            
            # Write correction function
            f.write("def correct_measurement(raw_measurement):\n")
            f.write('    """\n')
            f.write("    Apply calibration correction to a raw LIDAR measurement.\n\n")
            f.write("    Args:\n")
            f.write("        raw_measurement: Raw distance measurement in meters\n\n")
            f.write("    Returns:\n")
            f.write("        float: Calibrated distance measurement in meters\n")
            f.write('    """\n')
            
            # Polynomial computation
            if self.degree == 1:
                f.write("    return coefficients[0] * raw_measurement + coefficients[1]\n\n")
            else:
                f.write("    # Using numpy's polyval for stable polynomial evaluation\n")
                f.write("    return np.polyval(coefficients, raw_measurement)\n\n")
            
            # Add a batch correction function
            f.write("def correct_measurements(raw_measurements):\n")
            f.write('    """\n')
            f.write("    Apply calibration correction to multiple raw measurements.\n\n")
            f.write("    Args:\n")
            f.write("        raw_measurements: List or array of raw distance measurements in meters\n\n")
            f.write("    Returns:\n")
            f.write("        numpy.ndarray: Calibrated distance measurements in meters\n")
            f.write('    """\n')
            f.write("    return np.polyval(coefficients, raw_measurements)\n")
    
    def save_plot(self, filename, actual_distances=None, measured_distances=None):
        """
        Save a plot of the calibration curve.
        
        Args:
            filename: Output file path
            actual_distances: Optional array of actual distances used for calibration
            measured_distances: Optional array of measured distances used for calibration
        """
        plt.figure(figsize=(10, 6))
        plt.title(f"Calibration Model - Degree {self.degree} Polynomial" + 
                 (f" (RMSE: {self.rmse:.2f} mm)" if self.rmse is not None else ""))
        plt.xlabel("Raw Measurement (m)")
        plt.ylabel("Calibrated Measurement (m)")
        plt.grid(True)
        
        # Plot the ideal line
        plt.plot([self.start_dist, self.end_dist], [self.start_dist, self.end_dist], 
                'k--', label="Ideal (y=x)")
        
        # Plot the raw data points if provided
        if actual_distances is not None and measured_distances is not None:
            plt.scatter(measured_distances, actual_distances, color='blue', label='Calibration Points')
        
        # Plot the calibration curve
        x_dense = np.linspace(self.start_dist * 0.9, self.end_dist * 1.1, 100)
        y_dense = np.polyval(self.coefficients, x_dense)
        plt.plot(x_dense, y_dense, 'r-', label=f"Calibration Curve (Degree {self.degree})")
        
        plt.legend()
        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

def generate_calibration_models(actual_distances, measured_distances, output_prefix, degrees=[1, 2, 3]):
    """
    Generate calibration models for multiple polynomial degrees.
    
    Args:
        actual_distances: Array of actual distances
        measured_distances: Array of measured distances
        output_prefix: Prefix for output files
        degrees: List of polynomial degrees to generate
        
    Returns:
        dict: Dictionary of CalibrationModel objects by degree
    """
    models = {}
    start_dist = min(actual_distances)
    end_dist = max(actual_distances)
    
    for degree in degrees:
        # Fit polynomial 
        coeffs = np.polyfit(measured_distances, actual_distances, degree)
        
        # Calculate corrected values and errors
        corrected = np.polyval(coeffs, measured_distances)
        errors = (corrected - actual_distances) * 1000  # Error in mm
        rmse = np.sqrt(np.mean(np.square(errors)))
        
        # Create model
        model = CalibrationModel(coeffs, degree, start_dist, end_dist, rmse)
        models[degree] = model
        
        # Save model file
        model_file = f"{output_prefix}_model_deg{degree}.py"
        model.save_to_file(model_file)
        
        # Save plot
        plot_file = f"{output_prefix}_model_deg{degree}.png"
        model.save_plot(plot_file, actual_distances, measured_distances)
        
        print(f"Degree {degree} polynomial calibration model:")
        print(f"  Coefficients: {coeffs}")
        print(f"  RMSE: {rmse:.2f} mm")
        print(f"  Model saved to {model_file}")
        print(f"  Plot saved to {plot_file}")
    
    return models

def load_calibration_model(file_path):
    """
    Load a calibration model from a Python file.
    
    Args:
        file_path: Path to calibration model file
        
    Returns:
        CalibrationModel object or None if loading fails
    """
    try:
        import importlib.util
        
        # Load module from file path
        spec = importlib.util.spec_from_file_location("calibration", file_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        
        # Extract coefficients
        coefficients = module.coefficients
        
        # Determine degree from coefficients length
        degree = len(coefficients) - 1
        
        # Try to extract range information from docstring
        start_dist = 0.0
        end_dist = 0.0
        rmse = None
        
        if module.__doc__:
            for line in module.__doc__.splitlines():
                if "Range:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        range_str = parts[1].strip()
                        range_parts = range_str.split("to")
                        if len(range_parts) == 2:
                            try:
                                start_dist = float(range_parts[0].strip().rstrip('m'))
                                end_dist = float(range_parts[1].strip().rstrip('m'))
                            except:
                                pass
                elif "RMSE:" in line:
                    parts = line.split(":")
                    if len(parts) > 1:
                        try:
                            rmse_str = parts[1].strip().split()[0]
                            rmse = float(rmse_str)
                        except:
                            pass
        
        # Create and return model
        return CalibrationModel(coefficients, degree, start_dist, end_dist, rmse)
    
    except Exception as e:
        print(f"Error loading calibration model: {e}")
        return None

# If run directly, perform a simple test
if __name__ == "__main__":
    # Create a simple test model
    test_coeffs = [1.05, -0.01]  # Simple linear correction
    model = CalibrationModel(test_coeffs, 1, 0.1, 0.5)
    
    # Test correction
    raw = 0.35
    corrected = model.correct_measurement(raw)
    print(f"Raw: {raw}m, Corrected: {corrected}m")
    
    # Test batch correction
    raw_batch = np.array([0.25, 0.35, 0.45])
    corrected_batch = model.correct_measurements(raw_batch)
    print(f"Raw batch: {raw_batch}, Corrected batch: {corrected_batch}")
