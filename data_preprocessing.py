import tensorflow
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_data_path='dataset/train/'
validation_data_path='dataset/val/'
test_data_path = 'dataset/test/'
def preprocess_data():
    train_generator = ImageDataGenerator(rescale=1./255.0,
                                    horizontal_flip=True,
                                    shear_range=0.1 ,
                                    zoom_range=0.1,
                                    height_shift_range=0.1,
                                    fill_mode='nearest')
    validation_generator=ImageDataGenerator(rescale=1.0/255.0)
    test_generator=ImageDataGenerator(rescale=1.0/255.0)
    
    train_data= train_generator.flow_from_directory(train_data_path, 
                                                    batch_size=32, 
                                                    class_mode='categorical',
                                                    target_size=(224, 224))

    validation_data = validation_generator.flow_from_directory(validation_data_path,
                                                             batch_size=32, 
                                                             shuffle=True, 
                                                             class_mode='categorical', 
                                                             target_size=(224, 224))
    test_data = test_generator.flow_from_directory(test_data_path, 
                                                   batch_size=32, 
                                                   shuffle=False, 
                                                   class_mode='categorical',
                                                   target_size=(224, 224))

    return train_data, validation_data, test_data
