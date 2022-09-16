from pydoc import classname
import cv2
import numpy as np
from PIL import Image
import tensorflow
from tensorflow.keras.models import load_model
from PIL import Image
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from matplotlib import pyplot as plt
import face_detection_for_inference

classnames =['Afridi','Mashrafe','Messi','Ronaldo', 'Will_Smith']

def prepare_image(image_name):
    img = img_to_array(image_name)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('float32')
    img = img / 255.0
    return img

def model_load(model_path):
    model = load_model(model_path)
    return model

def inference_image(original_image,face_image, model):
    original_image=cv2.imread(original_image,1)              #read the original image
    original_image = cv2.resize(original_image, (400,400))

    face_image=cv2.imread(face_image,1)                      # read the images after face_detection
    
    cv2.imshow('original_image',original_image)
    cv2.imshow('only_face',face_image)
    cv2.waitKey(0)

    img =prepare_image(face_image)                  # prepare_image function is used for converting the image into array, reshape, and rescale it

    predict_value =model.predict(img)               
    index =np.argmax(predict_value)
    print("\n\n")
    print('The Name of given image is : ')
    print(classnames[index])


#

def process_image(src_dir,dest_dir, height,width):
    detect_face = face_detection_for_inference.align_crop_resize(src_dir,dest_dir,height,width)  # return true or false after performing face detection
    return detect_face

if __name__=="__main__":
    #model_name = 'models/my_model.h5'              # Load customize model     
    model_name = 'models/my_model_pre_trained.h5'   # Load the pre_trained model for inference
    src_dir='imges_for_inference/original_images'   # The source directory for the imge which have to be inferenced
    dest_dir='imges_for_inference/processed_image'  # The directory where the image will be saved after face detection
    height=224
    width=224
    face_detection = process_image(src_dir,dest_dir,height,width)        # process_image function is called to perform face_detection
    
    if face_detection:                                                   # if face detection is possible
        original_image='test100.jpg'       # Location of the image which have to be inferred
        face_image='imges_for_inference/processed_image/Cropped Image/test100.jpg' # Location of the images after performing face detection

        model = model_load(model_name)                      # load the model
        inference_image(original_image,face_image, model)  # inference_image function will perform inference for the detected face image
    else:
        print('Face detection is not performed so face recognition is not possible')



