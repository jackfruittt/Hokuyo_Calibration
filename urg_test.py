#!/usr/bin/env python3
"""
Hokuyo UTM-30LX Test Script Using URG Driver

A simple script to test the urg.py driver with a Hokuyo UTM-30LX LIDAR.

Usage:
python urg_test.py --port COM6
"""

import argparse
import time
import sys

# Import URG driver (urg.py must be in the same directory)
try:
    from pyurg import UrgDevice
except ImportError:
    print("Error: urg.py not found in the current directory.")
    print("Please ensure urg.py is in the same directory as this script.")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Hokuyo UTM-30LX Test Script')
    parser.add_argument('--port', type=str, default='COM6', 
                        help='Serial port (e.g., COM6, /dev/ttyACM0)')
    parser.add_argument('--duration', type=int, default=10, 
                        help='Duration to run in seconds (default: 10)')
    
    args = parser.parse_args()
    
    print("Hokuyo UTM-30LX Test Script (URG Driver)")
    print("=" * 50)
    
    try:
        # Connect to the LIDAR
        print(f"Connecting to LIDAR on {args.port}...")
        lidar = UrgDevice()
        if not lidar.connect(port=args.port):
            print("Failed to connect to LIDAR")
            return 1
        
        print("Successfully connected!")
        
        # Get sensor info
        info = lidar.get_sensor_info()
        print("\nSensor Information:")
        for key, value in info.items():
            print(f"  {key}: {value}")
        
        # Calculate center index (0 degrees)
        center_index = int(info.get('AFRT', 540))
        print(f"\nScan Information:")
        print(f"  Center index (0°): {center_index}")
        
        # Continuous measurement test
        print("\nStarting continuous measurement...")
        print("\nTime\t\tDistance (mm)\tDistance (m)")
        print("-" * 40)
        
        start_time = time.time()
        count = 0
        
        while time.time() - start_time < args.duration:
            # Get a scan
            data, timestamp = lidar.capture()
            
            # Extract the measurement at the center
            if data and len(data) > center_index:
                distance = data[center_index]
                elapsed = time.time() - start_time
                print(f"{elapsed:.2f}s\t\t{distance}\t\t{distance/1000:.3f}")
            else:
                print("Error: Invalid scan data")
            
            count += 1
            time.sleep(0.5)
        
        # Close the connection
        try:
            lidar.laser_off()
            lidar.close()
        except:
            pass
            
        print(f"\nTest complete. Performed {count} measurements over {time.time() - start_time:.2f} seconds.")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0

if __name__ == "__main__":
    sys.exit(main())