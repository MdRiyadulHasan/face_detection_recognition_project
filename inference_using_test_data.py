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
import face_detection_all_operations

classnames=['Afridi','Mashrafe','Messi','Ronaldo', 'Will_Smith']

# this code used for inference a image using test data set

def model_load(model_path):
    model = load_model(model_path)
    return model
def prepare_image(filename):
    #img = load_img(filename, target_size=(224, 224))
    img = img_to_array(filename)
    img = img.reshape(1, img.shape[0], img.shape[1], img.shape[2])
    img = img.astype('float32')
    img = img / 255.0
    return img


if __name__=="__main__":

    model_name = 'models/my_model_pre_trained.h5' 

    filename='dataset/test/Will_Smith/Willsmith.96.jpeg'   # the image location of a test data 
    img=cv2.imread(filename,1)
    cv2.imshow('inference_image',img)
    cv2.waitKey(0) 

    model = model_load(model_name)
    img = prepare_image(img)           # prepare_image function will convert the image into array, reshape and rescale it

    predict_value =model.predict(img)               
    index =np.argmax(predict_value)
    print("\n\n")
    
    print('The Name of given image is : ')
    print(classnames[index])

