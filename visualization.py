#!/usr/bin/env python3
"""
Visualization Module for Hokuyo UTM-30LX LIDAR Calibration

This module provides visualization functionality for both
calibration data and real-time LIDAR measurements.
"""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import os

class CalibrationVisualizer:
    """Class for visualizing calibration data."""
    
    def __init__(self, start_dist, end_dist):
        """
        Initialize the calibration visualizer.
        
        Args:
            start_dist: Calibration range start (meters)
            end_dist: Calibration range end (meters)
        """
        self.start_dist = start_dist
        self.end_dist = end_dist
        self.fig = None
        self.ax = None
        self.ax_error = None
    
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
    
    def update_plot(self, data):
        """
        Update the calibration plot with current data.
        
        Args:
            data: List of calibration data points
        """
        if not self.fig:
            self.initialize_plot()
        
        # Prepare the data
        actual_distances = [item[0] for item in data]
        measured_means = [item[2] for item in data]
        measured_stds = [item[3] for item in data]
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
    
    def show_plot_blocking(self):
        """Show the plot in blocking mode until the user closes it."""
        plt.ioff()
        plt.show(block=True)

class LidarVisualizer:
    """Class for real-time visualization of LIDAR data."""
    
    def __init__(self, lidar, calibration_model=None, max_range=5.0, polar_view=True):
        """
        Initialize the LIDAR visualizer.
        
        Args:
            lidar: LIDAR device object
            calibration_model: Optional calibration model for correction
            max_range: Maximum range to display in meters
            polar_view: Whether to use polar coordinates for visualization
        """
        self.lidar = lidar
        self.calibration_model = calibration_model
        self.max_range = max_range
        self.polar_view = polar_view
        
        # Determine center index for 0 degrees
        if hasattr(self.lidar, 'get_sensor_info'):
            info = self.lidar.get_sensor_info()
            if 'AFRT' in info:
                self.center_index = int(info['AFRT'])
            else:
                self.center_index = self.get_scan_size() // 2
        else:
            self.center_index = self.get_scan_size() // 2
        
        # Initialize figure and axes
        if self.polar_view:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7), subplot_kw={'polar': True})
            self.raw_ax, self.cal_ax = self.axes
            self.raw_ax.set_title("Raw Data")
            self.cal_ax.set_title("Calibrated Data")
            
            # Set the radial limit
            self.raw_ax.set_rlim(0, self.max_range)
            self.cal_ax.set_rlim(0, self.max_range)
            
            # Initialize empty plots
            self.raw_line, = self.raw_ax.plot([], [], 'r.', markersize=2)
            self.cal_line, = self.cal_ax.plot([], [], 'g.', markersize=2)
        else:
            self.fig, self.axes = plt.subplots(1, 2, figsize=(14, 7))
            self.raw_ax, self.cal_ax = self.axes
            self.raw_ax.set_title("Raw Data")
            self.cal_ax.set_title("Calibrated Data")
            
            # Set the axis limits
            self.raw_ax.set_xlim(-self.max_range, self.max_range)
            self.raw_ax.set_ylim(-1, self.max_range)
            self.cal_ax.set_xlim(-self.max_range, self.max_range)
            self.cal_ax.set_ylim(-1, self.max_range)
            
            # Set aspect ratio
            self.raw_ax.set_aspect('equal')
            self.cal_ax.set_aspect('equal')
            
            # Initialize empty plots
            self.raw_line, = self.raw_ax.plot([], [], 'r.', markersize=2)
            self.cal_line, = self.cal_ax.plot([], [], 'g.', markersize=2)
        
        # Add LIDAR position indicator
        if not self.polar_view:
            self.raw_ax.plot([0], [0], 'bo', markersize=8)
            self.cal_ax.plot([0], [0], 'bo', markersize=8)
    
    def get_scan_size(self):
        """Get the size of a full scan."""
        if hasattr(self.lidar, 'get_scan_lines'):
            return self.lidar.get_scan_lines()
        else:
            # Try to get a scan and determine size
            try:
                if hasattr(self.lidar, 'get_distance'):
                    data, _ = self.lidar.get_distance()
                elif hasattr(self.lidar, 'capture'):
                    data, _ = self.lidar.capture()
                return len(data)
            except:
                return 1081  # Default for UTM-30LX
    
    def animate(self, i):
        """
        Animation function for updating the plot.
        
        Args:
            i: Animation frame index
            
        Returns:
            Tuple of line objects that were updated
        """
        try:
            # Get a full scan
            if hasattr(self.lidar, 'get_distance'):
                data, _ = self.lidar.get_distance()
            elif hasattr(self.lidar, 'capture'):
                data, _ = self.lidar.capture()
            else:
                return self.raw_line, self.cal_line
            
            if not data:
                return self.raw_line, self.cal_line
            
            # Convert distances to meters
            distances = np.array(data) / 1000.0
            
            # Filter out invalid measurements
            valid_indices = distances > 0.01
            distances = distances[valid_indices]
            
            # Get angle indices for valid measurements
            angle_indices = np.arange(len(data))[valid_indices]
            angles = np.array([self.get_angle(idx) for idx in angle_indices])
            
            # Apply calibration if available
            if self.calibration_model:
                calibrated_distances = self.calibration_model.correct_measurements(distances)
            else:
                calibrated_distances = distances.copy()
            
            # Limit distances to max range
            distances = np.minimum(distances, self.max_range)
            calibrated_distances = np.minimum(calibrated_distances, self.max_range)
            
            if self.polar_view:
                # Polar coordinates
                self.raw_line.set_data(angles, distances)
                self.cal_line.set_data(angles, calibrated_distances)
            else:
                # Cartesian coordinates
                x_raw = distances * np.cos(angles)
                y_raw = distances * np.sin(angles)
                x_cal = calibrated_distances * np.cos(angles)
                y_cal = calibrated_distances * np.sin(angles)
                
                self.raw_line.set_data(x_raw, y_raw)
                self.cal_line.set_data(x_cal, y_cal)
            
            return self.raw_line, self.cal_line
        
        except Exception as e:
            print(f"Error in animation: {e}")
            return self.raw_line, self.cal_line
    
    def get_angle(self, index):
        """
        Convert index to angle in radians.
        
        Args:
            index: Measurement index
            
        Returns:
            float: Angle in radians
        """
        if hasattr(self.lidar, 'index2rad'):
            return self.lidar.index2rad(index)
        else:
            # Default conversion for UTM-30LX
            return (index - self.center_index) * (2.0 * np.pi / 1080)
    
    def run(self):
        """Run the visualizer."""
        # Set up the animation
        ani = animation.FuncAnimation(
            self.fig, self.animate, interval=100, blit=True)
        
        # Add title with calibration info
        if self.calibration_model:
            cal_name = f"Degree {self.calibration_model.degree} (RMSE: {self.calibration_model.rmse:.2f} mm)"
            plt.suptitle(f"Hokuyo UTM-30LX LIDAR Visualization (Calibration: {cal_name})")
        else:
            plt.suptitle("Hokuyo UTM-30LX LIDAR Visualization (No Calibration)")
        
        # Show the plot
        plt.tight_layout()
        plt.show()
        
        return True