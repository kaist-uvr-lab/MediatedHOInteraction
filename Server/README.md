## Installation
### Requirements
- Python-3.8
- CUDA 11.7
- requirements.txt

## Setup - HandTracker

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