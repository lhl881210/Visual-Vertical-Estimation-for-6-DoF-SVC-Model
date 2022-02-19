# Visual-Vertical-Estimation-for-6-DoF-SVC-Model

This is a very simple image processing method of visual vertical (VV) estimation for 6 DoF SVC Model.

## Paper
<img width="555" alt="image" src="https://user-images.githubusercontent.com/15242269/154702863-8b8a9a42-6a54-41da-8eb0-8caa96d4ab54.png">
<img width="655" alt="image" src="https://user-images.githubusercontent.com/15242269/154703081-87a4c189-063b-4533-aed8-81edb068ec24.png">

### arXiv
https://arxiv.org/abs/2202.06299

### ResearchGate
https://www.researchgate.net/publication/358703507

### BibTeX
```bash
@misc{liu2022motion,
    title={Motion Sickness Modeling with Visual Vertical Estimation and Its Application to Autonomous Personal Mobility Vehicles},
    author={Hailong Liu and Shota Inoue and Takahiro Wada},
    year={2022},
    eprint={2202.06299},
    archivePrefix={arXiv},
    primaryClass={cs.HC}
}
```
### Algorithm
<img width="644" alt="image" src="https://user-images.githubusercontent.com/15242269/154702974-87a3beb8-6234-4c19-8ac7-124618e99d6e.png">

## Python packages request

```bash
import matplotlib.pyplot as plt
import cv2
import numpy as np
import pandas as pd
import sys,getopt
import datetime
```

## Usage
```bash
$ python VV_for_SVC.py -i <inputfile> --camera <True/False> --camera_port <port number> --scale <1>
```
-i: input file.

--camera True: if using a camera.

--camera_port 0: if the camera port is 0.

--scale 2: if you want to change the image resolution to 1/2 of the original resolution. (even number is recommended)

### Exmple 1: input a video file (mp4 or avi)
```bash
$ python VV_for_SVC.py -i test.mp4 --scale 1
```

### Exmple 2: using a camera with port 0
```bash
$ python VV_for_SVC.py --camera True --camera_port 0 --scale 1
```
### NOTE
Please stop the program by pushing the "ESC" key!!!

If not, the estemated VV data will NOT BE SAVED in a CSV file when you using a CAMERA!!!

### Output
#### The estemated VV data in a CSV file
The CSV file includes following information:

[frame number, VV_acc_x[m/s^2]，VV_acc_y[m/s^2]，VV_acc_rad，VV_acc_dig]

The angle of VV is in [30,150] degree.

#### MP4 file
<img width="639" alt="image" src="https://user-images.githubusercontent.com/15242269/154701599-9af8bea4-ffaa-4c89-94e0-d5971227189e.png">

##### Upper left image: 
Input image. the estmated angle of VV is shown by green text.

##### Upper right image: 
The stabilized image by the estmated angle of VV.

##### Middle left image: 
The gradient’s magnitudes.

##### Middle right image: 
The gradient’s magnitudes after a filter.

##### Lower image: 
Histgram of gradient’s angles.

