import cv2
from concurrent.futures import process
import data_preprocessing
import numpy as np
import confusion_matrix_classification_report
import tensorflow
from tensorflow.keras.models import load_model


if __name__ == '__main__':
    train_data, validation_data, test_data = data_preprocessing.preprocess_data()    # Loading train, test and validation data again to evaluate the performance
    print(test_data.class_indices)
    #model_name ='models/my_model.h5'             # This line used for load customize model
    model_name='models/my_model_pre_trained.h5'   #This line used for load pre_trained model
    model =load_model(model_name)

    # calculation of accuracy 
    _,acc=model.evaluate(test_data)
    print(f'Accuracy is {round(acc*100, 2)}')

    # The line given below is used for creating confusion_matrix and classification_report

    confusion_matrix_classification_report.showConfusion_matrix_classification_report(model,test_data)
    
 