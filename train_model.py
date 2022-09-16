from tabnanny import verbose
from sklearn import metrics
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


def model_training(model,train_data, validation_data):
    
    epochs=100
    batch_size=32
    train_data_length=len(train_data.labels)
    validation_data_length=len(validation_data.labels)
    steps_per_epoch=(train_data_length//batch_size)
    validation_steps=(validation_data_length//batch_size)

    checkpoint = ModelCheckpoint("models/my_model_pre_trained.h5",
                                  monitor='val_accuracy', 
                                  verbose=1,
                                  save_best_only=True,
                                  save_weights_only=False,
                                  mode='auto', period=1)
    #early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=5, verbose=1, mode='auto')   #  # In early_stopping (patience=7) for pre_trained model   
    early_stopping = EarlyStopping(monitor='val_accuracy', min_delta=0, patience=7, verbose=1, mode='auto')  # In early_stopping (patience=7) for customize model
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    history=model.fit(train_data, 
                     steps_per_epoch=steps_per_epoch,
                     validation_data=validation_data,
                     validation_steps=validation_steps,
                     epochs=epochs, 
                     callbacks=[checkpoint,early_stopping], 
                     verbose=1)
    print('model training completed')
    return history
