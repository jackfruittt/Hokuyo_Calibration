#!/usr/bin/env python3
"""
Hokuyo UTM-30LX Calibration Tool Using URG Driver

This script creates a calibration dataset for a Hokuyo UTM-30LX LIDAR
by collecting multiple samples at specific distance intervals and saving
the results for later calibration model building.

This version uses the fixed URG driver (urg.py must be in the same directory).

Requirements:
- Python 3.6+
- pyserial (install with: pip install pyserial)
- numpy (install with: pip install numpy)
- matplotlib (install with: pip install matplotlib)
- keyboard (install with: pip install keyboard)

Usage:
python urg_calibration.py --port COM6
"""

import argparse
import time
import numpy as np
import os
import sys
import csv
import matplotlib.pyplot as plt
from datetime import datetime
import keyboard

# Import URG driver (urg.py must be in the same directory)
try:
    from urg import UrgDevice
except ImportError:
    print("Error: urg.py not found in the current directory.")
    print("Please ensure urg.py is in the same directory as this script.")
    sys.exit(1)

class LidarCalibrationTool:
    """
    Tool for calibrating a Hokuyo UTM-30LX LIDAR using the URG driver.
    
    This class handles the connection to the LIDAR, data collection,
    and orchestration of the calibration process.
    """
    
    def __init__(self, port, output_file, start_dist=0.1, end_dist=0.3, 
                 increment=0.005, samples_per_point=30):
        """
        Initialize the calibration tool.
        
        Args:
            port: Serial port for the LIDAR
            output_file: CSV file to save calibration data
            start_dist: Starting distance in meters
            end_dist: Ending distance in meters
            increment: Distance increment in meters
            samples_per_point: Number of samples to collect at each distance
        """
        self.port = port
        self.output_file = output_file
        self.start_dist = start_dist
        self.end_dist = end_dist
        self.increment = increment
        self.samples_per_point = samples_per_point
        
        # Calculate calibration points
        self.calibration_points = np.arange(start_dist, end_dist + increment/2, increment)
        self.current_point_index = 0
        
        # Initialize data storage
        self.data = []
        self.lidar = None
        self.center_index = None
        
        # Plotting
        self.fig = None
        self.ax = None
        self.ax_error = None
    
    def connect(self):
        """
        Connect to the LIDAR and get basic info.
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            print(f"Connecting to Hokuyo LIDAR on {self.port}...")
            self.lidar = UrgDevice()
            
            if not self.lidar.connect(port=self.port):
                print("Failed to connect to LIDAR")
                return False
            
            # Get sensor info
            info = self.lidar.get_sensor_info()
            print("\nSensor Information:")
            for key, value in info.items():
                print(f"  {key}: {value}")
            
            # Calculate the center index (for 0 degrees measurement)
            # The front index is already provided in the parameters
            if 'AFRT' in info:
                self.center_index = int(info['AFRT'])
            else:
                # Fallback to estimating the center
                self.center_index = self.lidar.get_scan_lines() // 2
            
            print(f"\nScan Information:")
            print(f"  Total scan points: {self.lidar.get_scan_lines()}")
            print(f"  Center index (0°): {self.center_index}")
            
            return True
        except Exception as e:
            print(f"Error connecting to LIDAR: {e}")
            return False
    
    def get_single_measurement(self):
        """
        Get a single distance measurement at 0 degrees.
        
        Returns:
            float: Distance in meters or -1 on error
        """
        try:
            # Get a full scan
            data, timestamp = self.lidar.get_distance()
            
            # Extract the center measurement (0 degrees)
            if data and len(data) > self.center_index:
                # Convert to meters
                return data[self.center_index] / 1000.0
            
            return -1
        except Exception as e:
            print(f"Error getting measurement: {e}")
            return -1
    
    def collect_samples(self, actual_distance):
        """
        Collect multiple samples at the specified distance.
        
        Args:
            actual_distance: The known/measured distance in meters
            
        Returns:
            Tuple: (samples_list, mean, standard_deviation)
        """
        print(f"\nCollecting {self.samples_per_point} samples at {actual_distance:.3f}m...")
        samples = []
        
        for i in range(self.samples_per_point):
            measurement = self.get_single_measurement()
            if measurement > 0:
                samples.append(measurement)
                print(f"  Sample {i+1}/{self.samples_per_point}: {measurement:.4f}m")
            else:
                print(f"  Error getting sample {i+1}/{self.samples_per_point}")
            time.sleep(0.1)  # Short delay between samples
        
        # Calculate statistics
        if samples:
            mean = np.mean(samples)
            std_dev = np.std(samples)
            print(f"  Mean: {mean:.4f}m, StdDev: {std_dev:.4f}m")
            return samples, mean, std_dev
        
        return [], 0, 0
    
    def initialize_plot(self):
        """Initialize the calibration plot."""
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.ax.set_title("Hokuyo UTM-30LX Calibration Data")
        self.ax.set_xlabel("Actual Distance (m)")
        self.ax.set_ylabel("Measured Distance (m)")
        self.ax.grid(True)
        self.ax.plot([self.start_dist, self.end_dist], [self.start_dist, self.end_dist], 
                     'k--', label="Ideal")
        
        # Error sub-plot (difference)
        self.ax_error = self.ax.twinx()
        self.ax_error.set_ylabel("Error (mm)")
        
        plt.tight_layout()
        plt.ion()  # Interactive mode on
        plt.show()
    
    def update_plot(self):
        """Update the calibration plot with current data."""
        if not self.fig:
            self.initialize_plot()
        
        # Prepare the data
        actual_distances = [item[0] for item in self.data]
        measured_means = [item[2] for item in self.data]
        measured_stds = [item[3] for item in self.data]
        errors = [(m - a) * 1000 for a, m in zip(actual_distances, measured_means)]  # Error in mm
        
        # Clear and redraw
        self.ax.clear()
        self.ax_error.clear()
        
        # Plot the calibration curve
        self.ax.set_title("Hokuyo UTM-30LX Calibration Data")
        self.ax.set_xlabel("Actual Distance (m)")
        self.ax.set_ylabel("Measured Distance (m)")
        self.ax.grid(True)
        self.ax.errorbar(actual_distances, measured_means, yerr=measured_stds, 
                         fmt='o-', label="Calibration Data")
        self.ax.plot([self.start_dist, self.end_dist], [self.start_dist, self.end_dist], 
                     'k--', label="Ideal")
        
        # Plot the error
        self.ax_error.bar(actual_distances, errors, alpha=0.3, color='r', width=0.002)
        self.ax_error.set_ylabel("Error (mm)")
        
        # Add labels and legends
        self.ax.legend(loc="upper left")
        
        # Update the plot
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
    
    def save_data(self):
        """Save the calibration data to a CSV file."""
        if not self.data:
            print("No data to save.")
            return
        
        try:
            with open(self.output_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                writer.writerow([
                    "actual_distance_m", 
                    "sample_count", 
                    "measured_mean_m", 
                    "measured_stddev_m",
                    "error_mm"
                ])
                
                # Write data rows
                for item in self.data:
                    actual_dist = item[0]
                    sample_count = item[1]
                    measured_mean = item[2]
                    measured_stddev = item[3]
                    error_mm = (measured_mean - actual_dist) * 1000  # Error in mm
                    
                    writer.writerow([
                        f"{actual_dist:.4f}",
                        sample_count,
                        f"{measured_mean:.6f}",
                        f"{measured_stddev:.6f}",
                        f"{error_mm:.2f}"
                    ])
            
            print(f"\nCalibration data saved to {self.output_file}")
            
            # Also save the raw samples for more detailed analysis if needed
            raw_file = os.path.splitext(self.output_file)[0] + "_raw.csv"
            with open(raw_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                
                # Write header
                header = ["actual_distance_m"]
                for i in range(self.samples_per_point):
                    header.append(f"sample_{i+1}")
                writer.writerow(header)
                
                # Write data rows
                for item in self.data:
                    row = [f"{item[0]:.4f}"]
                    samples = item[4]  # Raw samples
                    
                    # Fill in available samples
                    for i in range(self.samples_per_point):
                        if i < len(samples):
                            row.append(f"{samples[i]:.6f}")
                        else:
                            row.append("")
                    
                    writer.writerow(row)
            
            print(f"Raw sample data saved to {raw_file}")
            
            # Generate a simple polynomial fit model for immediate use
            self.generate_calibration_model()
            
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def generate_calibration_model(self):
        """Generate a simple polynomial calibration model from the data."""
        if len(self.data) < 3:
            print("Not enough data points for calibration model.")
            return
        
        try:
            # Extract actual and measured distances
            actual_distances = np.array([item[0] for item in self.data])
            measured_means = np.array([item[2] for item in self.data])
            
            # Fit polynomial models of different degrees
            for degree in [1, 2, 3]:
                # Fit polynomial 
                coeffs = np.polyfit(measured_means, actual_distances, degree)
                poly = np.poly1d(coeffs)
                
                # Calculate corrected values and errors
                corrected = poly(measured_means)
                errors = (corrected - actual_distances) * 1000  # Error in mm
                rmse = np.sqrt(np.mean(np.square(errors)))
                
                print(f"\nDegree {degree} polynomial calibration model:")
                print(f"  Coefficients: {coeffs}")
                print(f"  RMSE: {rmse:.2f} mm")
                
                # Also create a plot with the calibration curve
                plt.figure(figsize=(10, 6))
                plt.title(f"Calibration Model - Degree {degree} Polynomial (RMSE: {rmse:.2f} mm)")
                plt.xlabel("Raw Measurement (m)")
                plt.ylabel("Calibrated Measurement (m)")
                plt.grid(True)
                
                # Plot the raw data points
                plt.scatter(measured_means, actual_distances, color='blue', label='Calibration Points')
                
                # Plot the ideal line
                plt.plot([self.start_dist, self.end_dist], [self.start_dist, self.end_dist], 
                         'k--', label="Ideal (y=x)")
                
                # Plot the calibration curve
                x_dense = np.linspace(min(measured_means) * 0.9, max(measured_means) * 1.1, 100)
                y_dense = poly(x_dense)
                plt.plot(x_dense, y_dense, 'r-', label=f"Calibration Curve (Degree {degree})")
                
                plt.legend()
                plt.tight_layout()
                
                # Save the plot
                plot_file = os.path.splitext(self.output_file)[0] + f"_model_deg{degree}.png"
                plt.savefig(plot_file)
                plt.close()
                print(f"  Calibration curve saved to {plot_file}")
                
                # Save the model
                model_file = os.path.splitext(self.output_file)[0] + f"_model_deg{degree}.py"
                with open(model_file, 'w') as f:
                    f.write('"""\n')
                    f.write(f"Hokuyo UTM-30LX Calibration Model (Degree {degree})\n")
                    f.write(f"Generated: {datetime.now()}\n")
                    f.write(f"Range: {self.start_dist:.3f}m to {self.end_dist:.3f}m\n")
                    f.write(f"RMSE: {rmse:.2f} mm\n")
                    f.write('"""\n\n')
                    
                    f.write("import numpy as np\n\n")
                    
                    # Write coefficients array
                    f.write("# Polynomial coefficients (highest degree first)\n")
                    f.write("coefficients = [\n")
                    for i, c in enumerate(coeffs):
                        f.write(f"    {c:.10e}")
                        if i < len(coeffs) - 1:
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
                    if degree == 1:
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
                
                print(f"  Model saved to {model_file}")
            
        except Exception as e:
            print(f"Error generating calibration model: {e}")
            import traceback
            traceback.print_exc()
    
    def run_calibration(self):
        """Run the interactive calibration process."""
        if not self.connect():
            print("Failed to connect to LIDAR. Exiting.")
            return
        
        try:
            self.initialize_plot()
            
            print("\n=== Hokuyo UTM-30LX Calibration Tool ===")
            print(f"Calibrating range: {self.start_dist:.3f}m to {self.end_dist:.3f}m")
            print(f"Increments: {self.increment*1000:.1f}mm ({len(self.calibration_points)} points)")
            print(f"Samples per point: {self.samples_per_point}")
            print("\nInstructions:")
            print("1. Position the LIDAR at exactly the displayed distance from a flat target")
            print("2. Press SPACEBAR to collect samples at that distance")
            print("3. Move to the next distance increment when prompted")
            print("4. Press 'q' at any time to quit and save the collected data")
            print("\nStarting calibration...\n")
            
            # Main calibration loop
            while self.current_point_index < len(self.calibration_points):
                actual_distance = self.calibration_points[self.current_point_index]
                
                print(f"\n>>> Position LIDAR at exactly {actual_distance:.3f}m ({actual_distance*1000:.1f}mm) <<<")
                print("Press SPACEBAR when ready to collect samples...")
                
                # Wait for spacebar
                while True:
                    if keyboard.is_pressed('q'):
                        print("\nCalibration aborted by user.")
                        self.save_data()
                        return
                    
                    if keyboard.is_pressed('space'):
                        # Wait for key release to avoid multiple triggers
                        while keyboard.is_pressed('space'):
                            time.sleep(0.01)
                        break
                    
                    time.sleep(0.1)
                
                # Collect samples at this distance
                samples, mean, std_dev = self.collect_samples(actual_distance)
                
                if samples:
                    # Store the data
                    self.data.append((
                        actual_distance,           # Actual distance
                        len(samples),              # Number of samples collected
                        mean,                      # Mean measured distance
                        std_dev,                   # Standard deviation
                        samples                    # Raw samples
                    ))
                    
                    # Update the plot
                    self.update_plot()
                    
                    # Move to next point
                    self.current_point_index += 1
                else:
                    print("Failed to collect samples. Try again.")
                
                # If we've completed all points, save the data
                if self.current_point_index >= len(self.calibration_points):
                    print("\nCalibration complete!")
                    self.save_data()
                    
                    # Keep the plot open until user closes it
                    print("\nPress any key to exit...")
                    plt.ioff()
                    plt.show(block=True)
                
        except Exception as e:
            print(f"Error during calibration: {e}")
            import traceback
            traceback.print_exc()
        finally:
            if self.lidar:
                try:
                    self.lidar.laser_off()
                    self.lidar.close()
                except:
                    pass
                print("LIDAR connection closed.")

def main():
    parser = argparse.ArgumentParser(description='Hokuyo UTM-30LX Calibration Tool')
    parser.add_argument('--port', type=str, default='COM6', 
                        help='Serial port (e.g., COM6, /dev/ttyACM0)')
    parser.add_argument('--output', type=str, default='lidar_calibration.csv',
                        help='Output CSV file for calibration data')
    parser.add_argument('--start', type=float, default=0.1,
                        help='Starting distance in meters (default: 0.1)')
    parser.add_argument('--end', type=float, default=0.3,
                        help='Ending distance in meters (default: 0.3)')
    parser.add_argument('--increment', type=float, default=0.005,
                        help='Distance increment in meters (default: 0.005 = 5mm)')
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of samples per distance point (default: 30)')
    
    args = parser.parse_args()
    
    print("Hokuyo UTM-30LX Calibration Tool (URG Driver Version)")
    print("=" * 50)
    
    calibration_tool = LidarCalibrationTool(
        port=args.port,
        output_file=args.output,
        start_dist=args.start,
        end_dist=args.end,
        increment=args.increment,
        samples_per_point=args.samples
    )
    
    calibration_tool.run_calibration()
    
    return 0

if __name__ == "__main__":
    main()
