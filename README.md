# MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer
Official implementation of the paper ['MonoATT: Online Monocular 3D Object Detection with Adaptive Token Transformer']

## Introduction
MonoATT achieves leading performance on KITTI *val* and *test* set. 



## Installation
1. Clone this project and create a conda environment:

2. Install pytorch and torchvision matching your CUDA version:
    ```
    conda install pytorch torchvision cudatoolkit
    ```
    
3. Install requirements and compile the deformable attention:
    ```
    pip install -r requirements.txt

    cd lib/models/monodetr/ops/
    bash make.sh
    
    cd ../../../..
    ```
    
4. Make dictionary for saving training losses:
    ```
    mkdir logs
    ```
 
5. Download [KITTI](http://www.cvlibs.net/datasets/kitti/eval_object.php?obj_benchmark=3d) datasets and prepare the directory structure as:
    ```
    │MonoDETR/
    ├──...
    ├──data/KITTIDataset/
    │   ├──ImageSets/
    │   ├──training/
    │   ├──testing/
    ├──...
    ```
    You can also change the data path at "dataset/root_dir" in `configs/monoatt.yaml`.
    
## Get Started

### Train
You can modify the settings of models and training in `configs/monoatt.yaml` and appoint the GPU in `train.sh`:

    bash train.sh configs/monoatt.yaml > logs/monoatt.log
   
### Test
The best checkpoint will be evaluated as default. You can change it at "tester/checkpoint" in `configs/monoatt.yaml`:

    bash test.sh configs/monoatt.yaml
