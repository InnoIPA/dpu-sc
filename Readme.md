<!--
 Copyright (c) 2022 Innodisk Crop.
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# dpu-sc 
![demo](doc/fig/billboard.png)
# Description
dpu-sc presented a rapid demo which run AI inference on DPU with MPSoC.

# Requirements
### Dependencies on platform
- Xilinx [KV260](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html)
  - Opencv
  - XIR
  - VART
  - Vitis-AI 1.4

### Python's requirements
```bash
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install scikit-build cmake opencv-python mock cython
sudo python3 -m pip install tensorflow==2.4.1 -f https://tf.kmtea.eu/whl/stable.html
```

# How to use dpusc
We provide three modes for AI sample:
1. customcnn: Default CNN model. For inference cats and dogs. In dpu-sc, you can add argument `-x cnn` to use it.
2. yolov3-voc: Default YOLO model. For inference some common objects. In dpu-sc, you can add argument `-x yolo` to use it.
3. License Plate Recognition(LPR): We supported taiwain plate license detection and recognition. Please replace the model path to `models/obj/yolov4_tiny_416_carplate_0606.xmodel` and anchor to `18,13,28,21,32,25,36,28,41,32,67,48` in config.json. You can add argument `-x yolo -lpr` to use it.

> Notice: Our models were built for DPU4096, if you want to use DPU3136 or others DPU config, please contact our PM James(james_chen@innodisk.com). Also, we supported Vitis-AI 1.4 now.

and if you want to change model, you can modify model path in config.json.
```bash 
python3 dpusc -i <path-to-image>        -x <xmodel-type>  -t <output-type>
              -v <path-to-video>
              -c <webcam device nodes>
```
## Example - CNN
```bash 
# Inference with image, output image, using CNN
python3 dpusc -i dataset/images/dog.jpg -x cnn -t image
```
After execute above command(use CNN xmodel), you will get the result and the output image like below:
![cnn-result-01](doc/fig/cnn-result-01.png)
![cnn-result-02](doc/fig/cnn-result-02.png)

## Example - YOLO
```bash   
# Inference with image, output DP, using yolo
python3 dpusc -i dataset/images/moto.jpg -x yolo -t dp
```
After execute above command(use YOLO xmodel), you will get the result and the output image like below:
![yolo-result-01](doc/fig/yolo-result-01.png)
![yolo-result-02](doc/fig/yolo-result-02.png)


## Example - LPR
```bash   
# Inference with image, output image, using LPR
python3 dpusc -i dataset/images/lpr-2.jpg -x yolo -lpr -t image
```
After execute above command(use LPR mode), you will get the result and the output image like below:
![lpr-result-01](doc/fig/lpr-result-01.png)
![lpr-result-02](doc/fig/lpr-result-02.png)

## Other Options
```bash 
# Inference with video, output image, using yolo
python3 dpusc -v dataset/videos/walking_humans.nv12.1920x1080.h264 -x yolo -t image

# Inference with video, output video, using yolo
python3 dpusc -v dataset/videos/walking_humans.nv12.1920x1080.h264 -x yolo -t video

# Inference with webcam, output DP, using yolo
python3 dpusc -c 0 -x yolo -t dp

# Inference with webcam, output image, using yolo
python3 dpusc -c 0 -x yolo -t image

# Inference with video, output DP, using LPR
python3 dpusc -v <video path> -x yolo -lpr -t dp

# Inference with video, output Video, using LPR
python3 dpusc -v <video path> -x yolo -lpr -t video

```

# Dataset rules
### If run with CNN, you must follow the format of dataset naming rule which is label on the prefix of file name.  
e.g.   
  - at images_demo we detect cat or dog.  
  - at images_usb we detect perfect or defect.  
  
and so on.  

# Config.json
Xmodel and environment settings are in `config.json`.
- DISPLAY
    ```json
    "DISPLAY": {
        "WIDTH": "1920",
        "HEIGHT": "1080"
    }
    ```
    |Key Name|Description|
    |:-|:-|
    |`WIDTH`|The width of your display resolution.|
    |`HEIGHT`|The height of your display resolution.|

- MODLES-XMODELS_CLASS
    ```json
    "MODLES": {
        "XMODELS_CLASS": {
            "TYPE": "cnn",
            "MODEL": "models/cnn/customcnn.xmodel",
            "CLASS": ["dog", "cat"],
            "INPUT_SIZE": [250, 200]
        }
    }
    ```
    |Key Name|Description|
    |:-|:-|
    |`TYPE`|Xmodel's type.|
    |`MODEL`|Path to xmodel.|
    |`CLASS`|The classes that the xmodel provide.|
    |`INPUT_SIZE`|The image size that the xmodel can accept.|
  
- MODLES-XMODELS_OBJ
    ```json
    "MODLES": {
        "XMODELS_OBJ": {
            "TYPE": "yolo",
            "MODEL": "models/obj/yolov3-voc.xmodel",
            "CLASS": ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",  "diningtable", "dog", "horse", "motobike", "person", "pottedplant", "sheep", "sofa", "train", "tv"],
            "ANCHORS": [ 10, 13, 16, 30, 33, 23, 30, 61, 62, 45, 59, 119, 116, 90, 156, 198, 373, 326],
            "INPUT_SIZE": [416, 416],
            "IOU": "0.213",
            "NMS": "0.45",
            "CONF": "0.2",
            "BOX_MAX_NUM": "30"
        }
    }
    ```
    |Key Name|Description|
    |:-|:-|
    |`TYPE`|Xmodel's type.|
    |`MODEL`|Path to xmodel.|
    |`CLASS`|The classes that the xmodel provide.|
    |`ANCHORS`|The anchors that the xmodel provide.|
    |`INPUT_SIZE`|The image size that the xmodel can accept.|
    |`IOU`|Xmodel's IoU(Intersection over Union).|
    |`NMS`|Xmodel's NMS(Non-Maximum Suppression).|
    |`CONF`|Xmodel's confidence.|
    |`BOX_MAX_NUM`|The maximum number of bounding box that can be displayed in an image.|
- OUTPUT
    ```json
    "OUTPUT": {
        "VIDEO_OUTPUT": "./output.mp4",
        "IMAGE_OUT_DIR": "./"
    }
    ```
    |Key Name|Description|
    |:-|:-|
    |`VIDEO_OUTPUT`|The path of the output video.|
    |`IMAGE_OUT_DIR`|The path of the output image directory.|

# Uint Test
provide unittest script in [/unittest](https://github.com/aiotads/DPU_SC/tree/main/unittest).

# FAQ
## Can't download and install tensorflow-2.4.1?
You can use following setps to download and install tensorflow or
use our RPM package to install (please contact james_chen@innodisk.com). 

1. Use following command to download the `tensorflow.whl`. 

    ```
    sudo wget https://github.com/KumaTea/tensorflow-aarch64/releases/download/v2.4/tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
    ```
2. Install the `tensorflow.whl` without any dependencies.
    ```
    sudo pip3 install --no-dependencies tensorflow-2.4.1-cp37-cp37m-linux_aarch64.whl
    ```
3. After install `tensorflow`, follow the instructions below to manually install dependencies which we need.
   
   1. Create a file named `requirements.txt`.
   2. Fill the following dependencies in requirements.txt.         
        ```
        Keras-Preprocessing==1.1.2
        flatbuffers==22.12.6
        termcolor==2.1.1
        astunparse==1.6.3
        gast==0.5.3
        opt-einsum==3.3.0
        typing-extensions==4.4.0
        wrapt==1.14.1
        google-api-python-client==2.70.0
        absl-py==1.3.0
        ```
   3. Use the following command to install the dependencies 
        ```
        python3 -m pip install -r requirements.txt
        ```
4. Now you can run the dpu-sc with tensorflow.

## Contribution
[Contributing](contributing.md)

## License
[MIT](LICENSE)
