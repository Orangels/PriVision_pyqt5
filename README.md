# PriVision
Computer Vision Interactive Platform based on PyQt by priv Lab


## Install
* **Environment:** [**`py-RFCN-priv`**](https://github.com/soeaver/py-RFCN-priv), [**`openpose`**](https://github.com/CMU-Perceptual-Computing-Lab/openpose), [**`PyOpenPose`**](https://github.com/FORTH-ModelBasedTracker/PyOpenPose)

1. clone this repository

    `git clone https://github.com/soeaver/PriVision.git` 
    
1. clone [**`pypriv`**](https://github.com/soeaver/pypriv)

    `git clone https://github.com/soeaver/pypriv.git` 
    
2. and modify line 13 of `pypriv/nnutils/caffeutils.py` to `/home/user/workspace/openpose-priv-dev/3rdparty/caffe/python`

3. install pyqt4:
    `sudo apt-get install pyqt4-dev-tools`
and darkstyle:
    `pip install qdarkstyle`

4. install python dependence：
`numpy>=1.13.2`
`scipy>=0.13.3`
`Pillow>=4.3.0`
`opencv>=3.3.0`
`easydict`
