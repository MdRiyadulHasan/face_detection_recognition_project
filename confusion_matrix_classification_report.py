from matplotlib import pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

classnames = ['Afridi', 'Mashrafe','Messi', 'Ronaldo', 'Will_Smith']

def showConfusion_matrix_classification_report(model,test_data):
    prediction_value=np.argmax(model.predict(test_data),axis=-1)
    print(test_data.class_indices)
    actual_value=test_data.classes

    #confusion Matrix
    cm =confusion_matrix(actual_value,prediction_value)
    print(cm)
    plt.figure(figsize=(7,7))
    sns.heatmap(cm, annot=True, cbar=False, xticklabels=classnames, yticklabels=classnames, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('True_value')
    plt.ylabel('Predicted_value')
    plt.savefig('figure/confusion_matrix_pre_trained.png', dpi=600)     # This line used for saving confusion matrix for pre_trained model
    #plt.savefig('figure/confusion_matrix_customized.png', dpi=600)       # This line used for saving confusion matrix for customized model
    
    plt.show()
    
    #classification_report
    print("\n Classification Report : \n")
    print(classification_report(actual_value,prediction_value))