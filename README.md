# Hokuyo UTM-30LX LIDAR Calibration Tools

A collection of Python tools for calibrating and visualizing data from a Hokuyo UTM-30LX LIDAR scanner.

## Features

- **Interactive Calibration**: Guided process for collecting calibration data at precise distances
- **Multi-degree Polynomial Models**: Generates linear, quadratic, and cubic calibration models
- **Real-time Visualization**: Shows both raw and calibrated LIDAR data side by side
- **SCIP 2.0 Protocol**: Direct implementation of the Hokuyo SCIP 2.0 protocol
- **Data Analysis**: Statistical analysis of calibration data with error visualization

## Installation

1. Clone or download this repository to your local machine.

2. Install the required dependencies:
   ```
   pip install numpy matplotlib pyserial keyboard
   ```
   install the URG driver for hokuyo LiDARs, link is here: https://sourceforge.net/p/urgnetwork/wiki/driver_en/

3. Ensure all Python files are in the same directory.

## Files

The project is organized into modules for better maintainability:

- **urg.py**: URG driver for Hokuyo LIDARs using SCIP 2.0 protocol
- **calibration_module.py**: Core calibration functionality (models, loading, saving)
- **data_collector.py**: Data collection functionality for calibration
- **visualization.py**: Visualization tools for both calibration and real-time data
- **calibration_tool.py**: Main script for running the calibration process
- **lidar_visualizer.py**: Real-time LIDAR data visualization with calibration

## Usage

### 1. Test the LIDAR connection:

First, make sure the LIDAR is properly connected and powered:

```bash
python urg_test.py --port COM6
```

Replace `COM6` with the appropriate serial port for your system.

### 2. Run the calibration process:

```bash
python calibration_tool.py --port COM6
```

This will guide you through collecting data at incremental distances. At each distance:
1. Position the LIDAR at the specified distance from a flat target
2. Press spacebar to collect samples
3. Move to the next position when prompted

### 3. Visualize LIDAR data with calibration:

After calibration is complete, visualize real-time data with the applied calibration:

```bash
python lidar_visualizer.py --port COM6 --calibration lidar_calibration_model_deg2.py
```

For a Cartesian view instead of polar:

```bash
python lidar_visualizer.py --port COM6 --calibration lidar_calibration_model_deg2.py --cartesian
```

## Calibration Process

The calibration process helps improve measurement accuracy by correcting systematic errors in the LIDAR's distance measurements. It works by:

1. Collecting multiple samples at known distances
2. Computing the relationship between actual and measured distances
3. Generating polynomial models that can correct new measurements

The recommended calibration range is 0.1m to 0.3m with 5mm increments, but this can be customized.

## Options

### Calibration Tool

```
--port PORT           Serial port (e.g., COM6, /dev/ttyACM0)
--output OUTPUT       Output file prefix for calibration data and models
--start START         Starting distance in meters (default: 0.1)
--end END             Ending distance in meters (default: 0.3)
--increment INCREMENT Distance increment in meters (default: 0.005 = 5mm)
--samples SAMPLES     Number of samples per distance point (default: 30)
```

### LIDAR Visualizer

```
--port PORT           Serial port (e.g., COM6, /dev/ttyACM0)
--calibration FILE    Path to calibration model file
--max-range RANGE     Maximum range to display in meters (default: 5.0)
--cartesian           Use cartesian coordinates instead of polar
```

## Using Calibration Models

The calibration models can be used in other projects to correct LIDAR measurements:

```python
# Import the calibration model
from lidar_calibration_model_deg2 import correct_measurement

# Get raw measurement from LIDAR (in meters)
raw_distance = 0.157

# Apply calibration
calibrated_distance = correct_measurement(raw_distance)
print(f"Raw: {raw_distance}m, Calibrated: {calibrated_distance}m")
```

For batch processing:

```python
from lidar_calibration_model_deg2 import correct_measurements
import numpy as np

# Array of raw measurements
raw_distances = np.array([0.157, 0.213, 0.278])

# Apply calibration to all measurements
calibrated_distances = correct_measurements(raw_distances)
```

## Hardware Setup

- Connect the Hokuyo UTM-30LX to your computer via USB
- Provide 12V DC power to the LIDAR (the USB connection is for data only)
- Ensure the LIDAR is firmly mounted on a stable surface
- Use a flat, perpendicular target surface for calibration

## Troubleshooting

If you encounter connection issues:

1. Verify the LIDAR is properly powered with 12V DC
2. Check that the USB cable is securely connected
3. Ensure no other software is using the COM port
4. Try restarting the LIDAR by disconnecting and reconnecting power
5. Verify you have the correct COM port number
