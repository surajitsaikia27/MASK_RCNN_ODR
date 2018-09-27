# Mask RCNN for object Detection in Videos and Images

# Step 1: create a conda virtual environment with python 3

```
conda create -n Maskrcnn pip python=3.6
```
In case if you don't have anaconda install then follow the following steps
```
sudo apt-get install curl
curl -O https://repo.continuum.io/archive/Anaconda3-5.0.1-Linux-x86_64.sh
sha256sum Anaconda3–5.0.1-Linux-x86_64.sh
bash Anaconda3–5.0.1-Linux-x86_64.sh
conda create -n Maskrcnn pip python=3.6
```
Step 2: Clone the Mask_RCNN_ODR repo and install the dependencies
The dependencies are included in the requirement.txt, and you can install them just by typing:
```
pip install -r requirements.txt
conda install -c menpo opencv3
```
Step 3: install pycocotools
```
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

Step 4: download the pre-trained weights
```
Go to the link https://github.com/matterport/Mask_RCNN/releases
download the mask_rcnn_coco.h5 file
place the file in the MASK_RCNN_ODR directory
```
Step 5:To detect objects in images:and videos type
```
python image_demo.py — path ‘path to the image directory’
python video_demo.py — path ‘path to the image directory’
```
