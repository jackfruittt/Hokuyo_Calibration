#!/usr/bin/env python3
"""
Simple Test Script for Hokuyo URG Driver

This script tests the basic functionality of the URG driver with
the Hokuyo UTM-30LX LIDAR and displays measurements at 0 degrees.

Usage:
python test_urg_driver.py --port COM6
"""

import argparse
import time
import sys

# Import URG library
try:
    from urg import UrgDevice
except ImportError:
    print("Error: URG library not found.")
    print("Please install the URG library from the Hokuyo documentation.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Hokuyo URG Driver Test')
    parser.add_argument('--port', type=str, default='COM6', 
                        help='Serial port (e.g., COM6, /dev/ttyACM0)')
    parser.add_argument('--duration', type=int, default=10, 
                        help='Test duration in seconds (default: 10)')
    
    args = parser.parse_args()
    
    print("Hokuyo UTM-30LX URG Driver Test")
    print("=" * 50)
    
    try:
        # Create and connect to the LIDAR
        print(f"Connecting to Hokuyo LIDAR on {args.port}...")
        lidar = UrgDevice()
        lidar.connect(args.port)
        print("Successfully connected!")
        
        # Get device information
        info = lidar.get_sensor_info()
        print("\nSensor Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Get the total number of scan lines
        scan_lines = lidar.get_scan_lines()
        front_index = scan_lines // 2  # Approximately the center of the scan
        
        # Print angle range
        min_angle = lidar.index2rad(0) * 180 / 3.14159
        max_angle = lidar.index2rad(scan_lines - 1) * 180 / 3.14159
        front_angle = lidar.index2rad(front_index) * 180 / 3.14159
        
        print(f"\nScan Information:")
        print(f"  Total scan points: {scan_lines}")
        print(f"  Angle range: {min_angle:.1f}° to {max_angle:.1f}°")
        print(f"  Front index: {front_index} (approx. {front_angle:.1f}°)")
        
        # Start measurement
        print("\nStarting measurements...")
        lidar.start_measurement()
        
        print("\nTime\t\tDistance (mm)\tDistance (m)")
        print("-" * 40)
        
        # Measure for specified duration
        start_time = time.time()
        while time.time() - start_time < args.duration:
            # Get distance data
            timestamp, data = lidar.get_distance()
            
            # Extract the measurement at the front index (0 degrees)
            if data and len(data) > front_index:
                front_distance = data[front_index]
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f}s\t\t{front_distance}\t\t{front_distance/1000:.3f}")
            else:
                print("Error: No data received")
            
            # Small delay
            time.sleep(0.5)
        
        # Stop measurement
        lidar.stop_measurement()
        
        # Disconnect
        lidar.disconnect()
        print("\nDisconnected from LIDAR")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())
