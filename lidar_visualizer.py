#!/usr/bin/env python3
"""
Hokuyo UTM-30LX LIDAR Visualizer

This script creates a real-time visualization of LIDAR data,
showing both raw and calibrated measurements.

Usage:
python lidar_visualizer.py --port COM6 --calibration lidar_calibration_model_deg2.py

The calibration file is optional. If not provided, only raw data will be shown.
"""

import argparse
import sys
import os

# Import modules
try:
    from pyurg import UrgDevice
    from calibration_module import load_calibration_model
    from visualization import LidarVisualizer
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all module files are in the same directory:")
    print("  - urg.py")
    print("  - calibration_module.py")
    print("  - visualization.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Hokuyo UTM-30LX LIDAR Visualizer')
    parser.add_argument('--port', type=str, default='COM6', 
                        help='Serial port (e.g., COM6, /dev/ttyACM0)')
    parser.add_argument('--calibration', type=str, default=None,
                        help='Path to calibration model file')
    parser.add_argument('--max-range', type=float, default=5.0,
                        help='Maximum range to display in meters (default: 5.0)')
    parser.add_argument('--cartesian', action='store_true',
                        help='Use cartesian coordinates instead of polar')
    
    args = parser.parse_args()
    
    print("Hokuyo UTM-30LX LIDAR Visualizer")
    print("=" * 50)
    
    try:
        # Connect to LIDAR
        print(f"Connecting to LIDAR on {args.port}...")
        lidar = UrgDevice()
        if not lidar.connect(port=args.port):
            print("Failed to connect to LIDAR. Exiting.")
            return 1
        
        print("Successfully connected to LIDAR!")
        
        # Load calibration model if specified
        calibration_model = None
        if args.calibration:
            print(f"Loading calibration model from {args.calibration}...")
            calibration_model = load_calibration_model(args.calibration)
            if calibration_model:
                print(f"Calibration model loaded: Degree {calibration_model.degree}, "
                      f"Range: {calibration_model.start_dist:.2f}m to {calibration_model.end_dist:.2f}m, "
                      f"RMSE: {calibration_model.rmse:.2f}mm")
            else:
                print("Failed to load calibration model.")
        
        # Initialize visualizer
        visualizer = LidarVisualizer(
            lidar=lidar,
            calibration_model=calibration_model,
            max_range=args.max_range,
            polar_view=not args.cartesian
        )
        
        # Run visualizer (blocking)
        visualizer.run()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        # Cleanup
        if 'lidar' in locals():
            try:
                lidar.laser_off()
                lidar.close()
            except:
                pass
            print("LIDAR connection closed.")
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
