#!/usr/bin/env python3
"""
Data Collection Module for Hokuyo UTM-30LX LIDAR Calibration

This module provides functionality for collecting calibration data
from the LIDAR at different distances.
"""

import time
import numpy as np
import csv
import os
import keyboard

class CalibrationDataCollector:
    """Class for collecting calibration data from a LIDAR."""
    
    def __init__(self, lidar, output_file, start_dist=0.1, end_dist=0.15, 
                 increment=0.005, samples_per_point=30):
        """
        Initialize the data collector.
        
        Args:
            lidar: LIDAR device object
            output_file: CSV file to save calibration data
            start_dist: Starting distance in meters
            end_dist: Ending distance in meters
            increment: Distance increment in meters
            samples_per_point: Number of samples to collect at each distance
        """
        self.lidar = lidar
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
        
        # Determine center index for 0 degrees measurement
        if hasattr(self.lidar, 'get_sensor_info'):
            info = self.lidar.get_sensor_info()
            if 'AFRT' in info:
                self.center_index = int(info['AFRT'])
            else:
                self.center_index = self.get_scan_size() // 2
        else:
            self.center_index = self.get_scan_size() // 2
    
    def get_scan_size(self):
        """Get the size of a full scan."""
        if hasattr(self.lidar, 'get_scan_lines'):
            return self.lidar.get_scan_lines()
        else:
            # Try to get a scan and determine size
            try:
                data, _ = self.lidar.get_distance()
                return len(data)
            except:
                return 1081  # Default for UTM-30LX
    
    def get_single_measurement(self):
        """
        Get a single distance measurement at 0 degrees.
        
        Returns:
            float: Distance in meters or -1 on error
        """
        try:
            # Get a full scan
            if hasattr(self.lidar, 'get_distance'):
                data, _ = self.lidar.get_distance()
            elif hasattr(self.lidar, 'capture'):
                data, _ = self.lidar.capture()
            else:
                print("Error: LIDAR object doesn't have a compatible measurement method")
                return -1
            
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
    
    def save_data(self):
        """Save the calibration data to CSV files."""
        if not self.data:
            print("No data to save.")
            return
        
        try:
            # Save summary data
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
            
            # Save raw samples
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
            
            return True
        except Exception as e:
            print(f"Error saving data: {e}")
            return False
    
    def collect_data_interactive(self, visualization_callback=None):
        """
        Run the interactive data collection process.
        
        Args:
            visualization_callback: Optional callback function for updating visualization
            
        Returns:
            bool: True if data collection was successful
        """
        print("\n=== Hokuyo UTM-30LX Calibration Data Collection ===")
        print(f"Calibrating range: {self.start_dist:.3f}m to {self.end_dist:.3f}m")
        print(f"Increments: {self.increment*1000:.1f}mm ({len(self.calibration_points)} points)")
        print(f"Samples per point: {self.samples_per_point}")
        print("\nInstructions:")
        print("1. Position the LIDAR at exactly the displayed distance from a flat target")
        print("2. Press SPACEBAR to collect samples at that distance")
        print("3. After collection, you can:")
        print("   - Press ENTER to accept and continue to next distance")
        print("   - Press R to repeat the measurement at current distance")
        print("   - Press B to go back to the previous distance")
        print("4. Press Q at any time to quit and save the collected data")
        print("5. Press ESC to cancel calibration without saving")
        print("\nStarting calibration...\n")
        
        # Main calibration loop
        while self.current_point_index < len(self.calibration_points):
            actual_distance = self.calibration_points[self.current_point_index]
            
            print(f"\n>>> Position LIDAR at exactly {actual_distance:.3f}m ({actual_distance*1000:.1f}mm) <<<")
            print(f"Measurement {self.current_point_index + 1} of {len(self.calibration_points)}")
            print("Press SPACEBAR when ready to collect samples...")
            
            # Wait for spacebar
            while True:
                if keyboard.is_pressed('q'):
                    print("\nCalibration paused. Do you want to save the data collected so far? (Y/N)")
                    time.sleep(0.5)  # Give time for key release
                    while True:
                        if keyboard.is_pressed('y'):
                            print("\nSaving collected data...")
                            self.save_data()
                            return True
                        if keyboard.is_pressed('n'):
                            print("\nCalibration aborted without saving.")
                            return False
                        time.sleep(0.1)
                
                if keyboard.is_pressed('escape'):
                    print("\nCalibration canceled without saving.")
                    return False
                
                if keyboard.is_pressed('space'):
                    # Wait for key release to avoid multiple triggers
                    while keyboard.is_pressed('space'):
                        time.sleep(0.01)
                    break
                
                time.sleep(0.1)
            
            # Collect samples at this distance
            samples, mean, std_dev = self.collect_samples(actual_distance)
            
            if samples:
                # Temporarily store the data
                current_data = (
                    actual_distance,           # Actual distance
                    len(samples),              # Number of samples collected
                    mean,                      # Mean measured distance
                    std_dev,                   # Standard deviation
                    samples                    # Raw samples
                )
                
                # Display the collected data
                print("\nCollected data:")
                print(f"  Mean: {mean:.4f}m, StdDev: {std_dev:.4f}m")
                print(f"  Error: {(mean - actual_distance) * 1000:.2f}mm")
                print("\nPress ENTER to accept and continue")
                print("Press R to repeat measurement at current distance")
                print("Press B to go back to previous measurement")
                print("Press Q to quit and save collected data")
                print("Press ESC to cancel without saving")
                
                # Wait for user decision
                decision_made = False
                while not decision_made:
                    if keyboard.is_pressed('enter'):
                        print("\nAccepted. Moving to next distance...")
                        # Store the data permanently
                        self.data.append(current_data)
                        
                        # Update visualization if callback provided
                        if visualization_callback:
                            visualization_callback(self.data)
                        
                        # Move to next point
                        self.current_point_index += 1
                        decision_made = True
                        time.sleep(0.5)  # Give time for key release
                    
                    elif keyboard.is_pressed('r'):
                        print("\nRepeating measurement...")
                        decision_made = True
                        time.sleep(0.5)  # Give time for key release
                    
                    elif keyboard.is_pressed('b'):
                        if self.current_point_index > 0:
                            self.current_point_index -= 1
                            print(f"\nGoing back to distance {self.calibration_points[self.current_point_index]:.3f}m")
                            # Remove the last data point if we're going back
                            if self.data and len(self.data) > 0:
                                self.data.pop()
                                # Update visualization if callback provided
                                if visualization_callback:
                                    visualization_callback(self.data)
                        else:
                            print("\nAlready at the first measurement point.")
                        decision_made = True
                        time.sleep(0.5)  # Give time for key release
                    
                    elif keyboard.is_pressed('q'):
                        print("\nSaving collected data and exiting...")
                        # Store the current data before quitting
                        self.data.append(current_data)
                        # Update visualization if callback provided
                        if visualization_callback:
                            visualization_callback(self.data)
                        self.save_data()
                        return True
                    
                    elif keyboard.is_pressed('escape'):
                        print("\nCalibration canceled without saving.")
                        return False
                    
                    time.sleep(0.1)
            else:
                print("Failed to collect samples. Try again.")
            
            # If we've completed all points, save the data
            if self.current_point_index >= len(self.calibration_points):
                print("\nCalibration complete!")
                self.save_data()
                return True
        
        return False
    
    def get_calibration_data(self):
        """
        Get the collected calibration data in a format suitable for model generation.
        
        Returns:
            tuple: (actual_distances, measured_means)
        """
        if not self.data:
            return np.array([]), np.array([])
        
        actual_distances = np.array([item[0] for item in self.data])
        measured_means = np.array([item[2] for item in self.data])
        
        return actual_distances, measured_means
