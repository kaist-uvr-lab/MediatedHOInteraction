# Mediated Hand-Object Interaction
- ARRC-UVR Wise UI 통합 프로젝트, Mediated Hand-Object Interaction.
- Git for Server-side. 
- Client(HMD)-side Unity project can be found in [WiseUI gitlab](https://gitlab.com/arrc-uvr/wiseui)

## Installation
### Requirements
- Python-3.8
- CUDA 11.7
- requirements.txt

### Setup - HandTracker

- Download pretrained model and Mano data (updated : 23/10/16)

```
https://www.dropbox.com/scl/fi/fs5rix3z5r7qi1bypfy2o/SAR_r5_AGCN4_cross_2layer_extraTrue_resnet34_Epochs50.zip?rlkey=cvw0rxlb0vavpipfq0j0ulpii&dl=0
```
```
https://www.dropbox.com/scl/fi/60hzlehmd74e2c3xo2pxz/mano.zip?rlkey=mrxkbn9yl06zmop6ml6n1ofsy&dl=0
```

- Locate the file at 
```
WISEUIServer/handtracker/checkpoint/[model_name]/checkpoint.pth
WISEUIServer/handtracker/mano_data/mano
```


- Check the path of pretrained model in `WISEUIServer/handtracker/config.py`, `checkpoint` parameter

- run 
```
activate [virtualenv]
cd ./WISEUIServer
python main.py
```

## Content
- Real-time Hand tracking and pre-defined gesture recognition with Hololens 2 input : `WISEUIServer/main.py` 


## Acknowledgement
- This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government(MSIT) (No.2019-0-01270, WISE AR UI/UX Platform Development for Smartglasses)

- The sensing data acquisition using HoloLens 2 Research Mode was implemented with reference to [this project](https://github.com/jdibenes/hl2ss/).





## Lisense
WiseUI Applications are released under a MIT license. For a list of all code/library dependencies (and associated licenses), please see Dependencies.md.(To be updated)

For a closed-source version of WiseUI's modules for commercial purposes, please contact the authors : uvrlab@kaist.ac.kr, ikbeomjeon@kaist.ac.kr



