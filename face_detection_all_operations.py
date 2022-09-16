from genericpath import isfile
import tensorflow
import cv2
import os
import numpy as np
from tqdm import tqdm
import shutil
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import face_detection_align_image
import face_detection_crop_image

# The codes given below is used to  detect the face of the dataset

def align_crop_resize(source_dir, dest_dir,height,width):
    aligned_dir = os.path.join(dest_dir, 'Aligned Images')
    cropped_dir =os.path.join(dest_dir, 'Cropped Image')
    if os.path.isdir(dest_dir):
        shutil.rmtree(dest_dir)
    os.mkdir(dest_dir)
    os.mkdir(aligned_dir)
    os.mkdir(cropped_dir)
    image_list =sorted(os.listdir(source_dir))
    success_count=0
    for i,file in enumerate(image_list):
         image_name=os.path.join(source_dir,file)
         if os.path.isfile(image_name):
            try:
                img=cv2.imread(image_name)
                shape=img.shape

                status, img = face_detection_align_image.align_image(img) #  this function is used for correcting the alignment using rotation
                if status:
                    aligned_path=os.path.join(aligned_dir,file)
                    cv2.imwrite(aligned_path, img)
                    cstatus, img=face_detection_crop_image.crop_image(img) # give image and crop_status after crop_image to find face
                    if cstatus:          
                        img=cv2.resize(img, (height, width))
                        cropped_path=os.path.join(cropped_dir, file)
                        cv2.imwrite(cropped_path, img)                # save the image into destination directory
                        success_count +=1
            except:
                print('file', image_name ,'face detection not possible')
    return success_count
