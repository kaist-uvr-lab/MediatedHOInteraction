# importing libraries
import os
import cv2
from PIL import Image
import numpy as np
import random
import tqdm
import natsort

# Folder which contains all the images
# from which video is to be generated
fourcc = cv2.VideoWriter_fourcc(*'DIVX')

def generate_video(image_list, vid_name, fps=15):
    image_folder = '.'  # make sure to use your folder
    video_name = vid_name

    frame = image_list[0]

    # setting the frame width, height width
    # the width, height of first image
    height, width, layer = frame.shape

    # cv2.VideoWriter(filename, fourcc, fps, frameSize, isColor=None)
    video = cv2.VideoWriter(video_name, fourcc, fps, (width, height))

    # Appending the images to the video one by one
    for idx, image in tqdm.tqdm(enumerate(image_list)):
        video.write(image)

    # Deallocating memories taken for window creation
    cv2.destroyAllWindows()
    video.release()  # releasing the video generated


target = 'vis_mp'   # vis_mp vis_ours
target_obj = 'app_0'    # cyl_0 key_0 app_0
target_path = f"C:/Woojin/research/MediatedHOInteraction/Integration/WiseUIServer/pkl_test/{target}/0/{target_obj}"


image_list = []
path_list = natsort.natsorted(os.listdir(target_path))
for file in path_list:
    # check file name is subj_cam_vis.png
    im = cv2.imread(os.path.join(target_path, file), cv2.IMREAD_UNCHANGED)
    image_list.append(im)

# Calling the generate_video function
generate_video(image_list, f'clip_{target}_{target_obj}' + '.avi')
