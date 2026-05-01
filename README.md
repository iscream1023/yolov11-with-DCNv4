# Integrate DCNv4 In YOLOv11 for Amorphous Object Detection

## Overview
This repository investigates the integration of Deformable Convolutional Networks (DCNv2[^1], DCNv4[^2]) into the YOLOv11 architecture to enhance the detection capabilities for amorphous objects, such as fire and smoke. 

## Dataset
The experiments were conducted using the **D-Fire dataset**[^3], a specialized image dataset constructed for fire and smoke detection tasks.

## Methodology
To evaluate the impact of deformable convolutions on extracting irregular and dynamic spatial features, we modified the baseline YOLOv11 architecture. Specifically, the **C2PSA** (Cross-Stage Partial Network with Spatial Attention) module in the baseline YOLOv11 was substituted with **DCNv2** and **DCNv4** modules.

## Experimental Results

### 1. Detection Performance
The following table presents the detection performance metrics evaluated on the YOLOv11-nano model.

| Metric | Baseline (YOLOv11) | Experiment 1 (YOLOv11 + DCNv2) | Experiment 2 (YOLOv11 + DCNv4) | Improvement (vs. Baseline) |
| :--- | :---: | :---: | :---: | :---: |
| mAP 50 | 0.7581 | 0.7613 | 0.7716 | +1.35%p |
| mAP 50-95 | 0.4328 | 0.4397 | 0.4471 | +1.43%p |
| Precision | 0.7583 | 0.7555 | 0.7602 | +0.19%p |
| Recall | 0.6791 | 0.6922 | 0.7030 | +2.39%p |

## Conclusion

Based on the experimental evaluations, we derive the following conclusions:
1. **Scale-dependent Performance Variation:** The integration of DCNv4 contributed to an improvement in detection accuracy for the YOLOv11-nano (`n`) model. However, no significant performance gains were observed when applied to the larger YOLOv11-small (`s`) model.
   
    | Metric | Baseline (YOLOv11s) | Experiment 3 (yolo 11s + DCNv4) |
    | :--- | :---: | :---: |
    | mAP 50 | 0.7865 | 0.7883 |
    | mAP 50-95 | 0.4610 | 0.4672 | 
    | Precision | 0.7757 | 0.7717 | 
    | Recall | 0.7215 | 0.7274 | 

2. **Hardware Deployment Constraints:** A critical limitation identified in this study is hardware deployability. Currently, deploying the modified architecture equipped with DCN onto NVIDIA Jetson edge boards presents significant compatibility and optimization challenges.

---

# Installation Guide

This repository provides a custom implementation of YOLOv11 integrated with DCNv4 (Deformable Convolution v4) to enhance detection performance, specifically optimized for handling complex object shapes and reducing CPU overhead.
Prerequisites

Ensure that you have CUDA installed.

This project is verified on: **CUDA 12.4 (Recommended) or CUDA 12.8**

Check your version in terminal using:

    ```
    nvcc --version
    ```
    
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


### Navigate to your local ultralytics directory
```
cd path/to/your/local/ultralytics
```
### Install in editable mode
```
pip install -e . --no-deps
```
### Step 4: Finalize Dependencies

Install any remaining required packages that may be missing in your environment.
Bash

### Install additional requirements
```
pip install -r requirements.txt
```

Key Technical Notes

    Editable Mode (-e): We use pip install -e . to link the virtual environment directly to the source code. This ensures that the custom DCNv4 layers in the nn module are correctly recognized by the Ultralytics engine.

    Dimension Alignment: The implementation handles the conversion between YOLO's BCHW format and DCNv4's expected BHWC format automatically within the Bottleneck_DCNv4 block.


## References

[^1]: Wang, R., Fu, B., Fu, G., & Wang, M. (2020). DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-scale Learning to Rank Systems. arXiv preprint arXiv:2008.13535. https://doi.org/10.48550/arXiv.2008.13535
[^2]: Lu, W., Jia, J., Li, R., He, J., & Dai, J. (2024). Efficient Deformable ConvNets: Rethinking Dynamic and Sparse Operator for Vision Applications. arXiv preprint arXiv:2401.06197. https://doi.org/10.48550/arXiv.2401.06197
[^3]: **D-Fire Dataset:** Venâncio, P. V. A. B., Lisboa, A. C., & Barbosa, A. V. (2022). An automatic fire detection system based on deep convolutional neural networks for low-power, resource-constrained devices. *Neural Computing and Applications*, 34(18), 15349-15368. https://doi.org/10.1007/s00521-022-07467-z
