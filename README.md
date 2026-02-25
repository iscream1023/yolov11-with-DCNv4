# YOLOv11 with DCNv4/DCNv2 Integration

This repository provides a custom implementation of YOLOv11 integrated with DCNv4 (Deformable Convolution v4) to enhance detection performance, specifically optimized for handling complex object shapes and reducing CPU overhead.
Prerequisites

Ensure that you have CUDA installed. This project is verified on:

    CUDA 12.4 (Recommended) or CUDA 12.8

    Check your version in terminal using: nvcc --version

# Installation Guide
### Step 1: Environment Setup

Create and activate a dedicated Conda virtual environment to ensure isolation.

```
conda create -n dcnv4_yolo python=3.10 -y
conda activate dcnv4_yolo
```

### Step 2: Install DCNv4

Download and build the DCNv4 package. Since DCNv4 requires local CUDA compilation, follow these steps:

Visit [DCNv4 PyPI](https://pypi.org/project/DCNv4/) on PyPI or clone the source.

Build and install within your activated environment:

### Example if building from source
```
conda activate dcnv4_yolo
cd rt/to/DCNv4
python setup.py install
```

### Step 3: Custom Ultralytics Setup

To use DCNv4/DCNv2 within the YOLO framework, you must use the customized nn modules provided in this repository.

    Overwrite the Module: Replace your local ultralytics/nn directory with the nn module provided in this GitHub repository.

    Editable Installation: Install the ultralytics package in editable mode. This allows you to modify the source code (like our DCNv4 integration) and have the changes reflected immediately without affecting other global installations.

Bash

### Navigate to your local ultralytics directory
cd path/to/your/local/ultralytics

### Install in editable mode
pip install -e . --no-deps

Step 4: Finalize Dependencies

Install any remaining required packages that may be missing in your environment.
Bash

### Install additional requirements
pip install -r requirements.txt

Key Technical Notes

    Editable Mode (-e): We use pip install -e . to link the virtual environment directly to the source code. This ensures that the custom DCNv4 layers in the nn module are correctly recognized by the Ultralytics engine.

    Dimension Alignment: The implementation handles the conversion between YOLO's BCHW format and DCNv4's expected BHWC format automatically within the Bottleneck_DCNv4 block.
