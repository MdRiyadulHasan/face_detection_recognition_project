import face_detection_all_operations
import dataset
import dataset
import data_preprocessing
import model
import train_model
import visualization

if __name__=='__main__':
    #dataset.create_dataset()
    train_data, validation_data, test_data = data_preprocessing.preprocess_data()
    #visualization.draw_some_train_data(train_data)
    total_class=len(train_data.class_indices)                  
    print(total_class)
    #model = model.define_customize_model(total_class)         # This line used for creating customize model
    model = model.define_pre_trained_model(total_class)                    # This line used for creating pre_trained model

    history =train_model.model_training(model,train_data, validation_data)
    loss_graph = visualization.train_validation_loss_graph(history)
    accuracy_graph=visualization.train_validation_accuracy_graph(history)
 
    
    # The codes given bellow was used to detect the faces from the images and once the face detection performed then there is no need to run again.
    
    
    """
    source_dir='images_face_detection/raw_images'
    dest_dir='images_face_detection/processed_image'
    height=224
    width=224
    count=face_detection_all_operations.align_crop_resize(source_dir,dest_dir,height,width)
    print ('Number of sucessfully processed images= ', count)

    """
    
    

    

    
    


    
    

    

    
  

   
   
    
   