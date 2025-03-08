#!/usr/bin/env python3
"""
Hokuyo UTM-30LX LIDAR Calibration Tool

This is the main script for the LIDAR calibration tool.
It brings together the modules for data collection, visualization,
and calibration model generation.

Usage:
python calibration_tool.py --port COM6
"""

import argparse
import sys
import os

# Import the other modules
try:
    from pyurg import UrgDevice
    from data_collector import CalibrationDataCollector
    from visualization import CalibrationVisualizer
    from calibration_module import generate_calibration_models
except ImportError as e:
    print(f"Error importing required modules: {e}")
    print("Make sure all module files are in the same directory:")
    print("  - urg.py")
    print("  - data_collector.py")
    print("  - visualization.py")
    print("  - calibration_module.py")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Hokuyo UTM-30LX Calibration Tool')
    parser.add_argument('--port', type=str, default='COM6', 
                        help='Serial port (e.g., COM6, /dev/ttyACM0)')
    parser.add_argument('--output', type=str, default='lidar_calibration',
                        help='Output file prefix for calibration data and models')
    parser.add_argument('--start', type=float, default=0.1,
                        help='Starting distance in meters (default: 0.1)')
    parser.add_argument('--end', type=float, default=0.3,
                        help='Ending distance in meters (default: 0.3)')
    parser.add_argument('--increment', type=float, default=0.005,
                        help='Distance increment in meters (default: 0.005 = 5mm)')
    parser.add_argument('--samples', type=int, default=30,
                        help='Number of samples per distance point (default: 30)')
    
    args = parser.parse_args()
    
    print("Hokuyo UTM-30LX LIDAR Calibration Tool")
    print("=" * 50)
    
    try:
        # Connect to LIDAR
        print(f"Connecting to LIDAR on {args.port}...")
        lidar = UrgDevice()
        if not lidar.connect(port=args.port):
            print("Failed to connect to LIDAR. Exiting.")
            return 1
        
        print("Successfully connected to LIDAR!")
        
        # Initialize visualizer
        visualizer = CalibrationVisualizer(args.start, args.end)
        
        # Initialize data collector
        output_csv = args.output + ".csv"
        collector = CalibrationDataCollector(
            lidar=lidar,
            output_file=output_csv,
            start_dist=args.start,
            end_dist=args.end,
            increment=args.increment,
            samples_per_point=args.samples
        )
        
        # Initialize plot
        visualizer.initialize_plot()
        
        # Collect data with visualization updates
        success = collector.collect_data_interactive(
            visualization_callback=visualizer.update_plot
        )
        
        if success:
            # Generate calibration models
            actual_distances, measured_means = collector.get_calibration_data()
            if len(actual_distances) > 0:
                models = generate_calibration_models(
                    actual_distances, 
                    measured_means, 
                    args.output
                )
                
                print("\nCalibration models generated successfully!")
                print("\nTo visualize LIDAR data with calibration:")
                print(f"python lidar_visualizer.py --port {args.port} --calibration {args.output}_model_deg2.py")
            else:
                print("No data collected. Calibration models not generated.")
            
            # Keep the plot open until user closes it
            print("\nClose the plot window when finished.")
            visualizer.show_plot_blocking()
        
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
