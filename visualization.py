import numpy as np
from matplotlib import pyplot as plt
from matplotlib.image import imread  

classnames = ['Afridi', 'Mashrafe','Messi', 'Ronaldo', 'Will_Smith']


def draw_some_train_data(train_data):
    x,y=train_data.next()
    plt.figure(figsize=(5,5))
    rows=3
    cols=3
    for i in range(rows*cols):
        plt.subplot(rows,cols,i+1)
        image=x[i]
        plt.imshow(image)
        label=np.argmax(y[i])
        plt.xlabel(classnames[label])
        plt.xticks([])
        plt.yticks([])
        plt.savefig('figure/train_dataset.png', dpi=600)                           # This line used for show some random train image
    plt.show()

# The code given below for drawing train and validation Loss graph

def train_validation_loss_graph(history):
    plt.figure(figsize=(8,8))
    plt.title('Train and validation loss graph')
    plt.plot(history.history['loss'], label='loss', color='blue')
    plt.plot(history.history['val_loss'], label='val_loss', color='orange')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Loss')
    #plt.savefig('figure/trainin_validation_loss_graph_112.png', dpi=600)   # This line used for saving accuracy graph for pre_trained model
    #plt.savefig('figure/trainin_validation_loss_graph_customize.png', dpi=600)    # This line used for saving accuracy graph for customized model
    plt.show()

# The code given below for drawing train and validation Accuracy graph

def train_validation_accuracy_graph(history):
    plt.figure(figsize=(8,8))
    plt.title('Train and validation accuracy graph')
    plt.plot(history.history['accuracy'], label='accuracy', color='blue')
    plt.plot(history.history['val_accuracy'], label='val_accuracy', color='orange')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('Accuracy')
    plt.savefig('figure/trainin_validation_accuracy_graph_112.png', dpi=600) 
    #plt.savefig('figure/trainin_validation_accuracy_graph_pre_trained.png', dpi=600)     # This line used for saving accuracy graph for pre_trained model
    #plt.savefig('figure/trainin_validation_accuracy_graph_customize.png', dpi=600)      # This line used for saving accuracy graph for customized model
    plt.show()