# Image Processing Project: Land Cover Classification with U-Net and GEE

This project implements a complete pipeline for land cover classification using Sentinel-1 and Sentinel-2 satellite imagery, Google Earth Engine (GEE), and a U-Net deep learning model.

## Project Structure

*   **`config.py`**: Central configuration file for all project parameters (paths, bands, dates, model settings). **Edit this file first!**
*   **`utils.py`**: Shared utility functions for GEE operations, data processing, and model inference.
*   **`requirements.txt`**: List of required Python packages.
*   **Fixed Dependencies**: Removed reliance on broken packages like `smooth-tiled-predictions-plus`.
*   **Automated Workflow**: Improved file handling and export logic.