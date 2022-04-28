<!--
 Copyright (c) 2022 Innodisk Crop.
 
 This software is released under the MIT License.
 https://opensource.org/licenses/MIT
-->

# DPU_SC 
![demo](dataset/images/street-2-out.png)
## Requirements
### python's requirements
```bash
sudo python3 -m pip install --upgrade pip
sudo python3 -m pip install scikit-build cmake opencv-python mock cython
sudo python3 -m pip install tensorflow==2.4.1 -f https://tf.kmtea.eu/whl/stable.html
```
### Platform
- Xilinx [KV260](https://www.xilinx.com/products/som/kria/kv260-vision-starter-kit.html) (config: DPU4096)
  - cv2
  - xir
  - vart
  - vitis-ai 1.4
# Run
> A PY wrap for xmodel runner.  
- #1 classification: cnn
- #2 object detection: yolo

| Type | model      | dataset      |
| ---- | :--------- | :----------- |
| #1   | customcnn  | cats and dog |
| #2   | yolov3-voc | voc data set |

```bash 
python3 dpusc -i <path-to-image>        -x <xmodel-type>  -t <output-type>
              -v <path-to-video>
              -c <webcam device nodes>
  
# Inference with image, output image, using cnn
python3 dpusc -i dataset/images/car.jpg -x cnn -t image

# Inference with image, output DP, using yolo
python3 dpusc -i dataset/images/car.jpg -x yolo -t dp

# Inference with video, output image, using yolo
python3 dpusc -v dataset/videos/walking_humans.nv12.1920x1080.h264 -x yolo -t image

# Inference with video, output video, using yolo
python3 dpusc -v dataset/videos/walking_humans.nv12.1920x1080.h264 -x yolo -t video

# Inference with webcam, output DP, using yolo
python3 dpusc -c 0 -x yolo -t dp

# Inference with webcam, output image, using yolo
python3 dpusc -c 0 -x yolo -t image
```

# Dataset rules
### If run with cnn, you must follow the format of dataset naming rule which is label on the prefix of file name.  
e.g.   
  - at images_demo we detect cat or dog.  
  - at images_usb we detect perfect or defect.  
  
and so on.  

# Config.json
```json
{
    "DISPLAY": {
        "WIDTH": "1920",
        "HEIGHT": "1080"
    },
    "MODLES": {
        "XMODELS_CLASS": {
            "TYPE": "cnn",
            "MODEL": "models/cnn/customcnn.xmodel",
            "CLASS": [
                "dog",
                "cat"
            ],
            "INPUT_SIZE": [
                250,
                200
            ]
        },
        "XMODELS_OBJ": {
            "TYPE": "yolo",
            "MODEL": "models/obj/yolov3-voc.xmodel",
            "CLASS": [
                "aeroplane",
                "bicycle",
                "bird",
                "boat",
                "bottle",
                "bus",
                "car",
                "cat",
                "chair",
                "cow",
                "diningtable",
                "dog",
                "horse",
                "motobike",
                "person",
                "pottedplant",
                "sheep",
                "sofa",
                "train",
                "tv"
            ],
            "ANCHORS": [
                10,
                13,
                16,
                30,
                33,
                23,
                30,
                61,
                62,
                45,
                59,
                119,
                116,
                90,
                156,
                198,
                373,
                326
            ],
            "INPUT_SIZE": [
                416,
                416
            ],
            "IOU": "0.213",
            "NMS": "0.45",
            "CONF": "0.2",
            "BOX_MAX_NUM": "30"
        }
    },
    "OUTPUT": {
        "VIDEO_OUTPUT": "./output.mp4",
        "IMAGE_OUT_DIR": "./"
    }
}
```
- If you want to change your display resolution, please modify `DISPLAY->WIDTH` and `DISPLAY->HEIGHT` in `config.json`.
- If you want to replace yolo xmodel, please modify below keys in `config.json`:

    | Key                                  | Description                           |
    | ------------------------------------ | :------------------------------------ |
    | MODLES -> XMODELS_OBJ -> MODEL       | Path to your xmodel.                  |
    | MODLES -> XMODELS_OBJ -> ANCHORS     | Your xmodel's anchors.                |
    | MODLES -> XMODELS_OBJ -> INPUT_SIZE  | Your xmodel's input size.             |
    | MODLES -> XMODELS_OBJ -> CONF        | Confidence score which you want.      |
    | MODLES -> XMODELS_OBJ -> BOX_MAX_NUM | The maximun number of bounding boxes. |
- If you want to change your image or video output directory, please modify `OUTPUT->VIDEO_OUTPUT` and `OUTPUT->IMAGE_OUT_DIR` in `config.json`.
# Uint Test
provide unittest script in [/unittest](https://github.com/aiotads/DPU_SC/tree/main/unittest).

## Contribution
[Contributing](contributing.md)

## License
[MIT](LICENSE)