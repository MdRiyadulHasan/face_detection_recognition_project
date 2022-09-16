import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import PIL
import glob

# The codes given below is used for plotting bar_plot of train, validation and test data.

# Bar_plot for train data

Afridi=glob.glob('dataset/train/Afridi/*.*')
Mashrafe=glob.glob('dataset/train/Mashrafe/*.*')
Messi=glob.glob('dataset/train/Messi/*.*')
Ronaldo=glob.glob('dataset/train/Ronaldo/*.*')
Will_Smith=glob.glob('dataset/train/Will_Smith/*.*')

classnames = ['Afridi','Mashrafe','Messi','Ronaldo','Will_Smith']
Numbers = [len(Afridi),len(Mashrafe),len(Messi),len(Ronaldo),len(Will_Smith)]
plt.bar(classnames,Numbers,color=['red', 'yellow', 'blue', 'black', 'orange'])
plt.title('Bar_plot of Train_dataset')
plt.savefig('figure/number_of_train_dataset.png',dpi=600)
plt.show()

# Bar_plot for test data

Afridi=glob.glob('dataset/test/Afridi/*.*')
Mashrafe=glob.glob('dataset/test/Mashrafe/*.*')
Messi=glob.glob('dataset/test/Messi/*.*')
Ronaldo=glob.glob('dataset/test/Ronaldo/*.*')
Will_Smith=glob.glob('dataset/test/Will_Smith/*.*')

classnames = ['Afridi','Mashrafe','Messi','Ronaldo','Will_Smith']
Numbers = [len(Afridi),len(Mashrafe),len(Messi),len(Ronaldo),len(Will_Smith)]
plt.bar(classnames,Numbers,color=['cyan', 'blue', 'red', 'green', 'yellow'])
plt.title('Bar_Plot of Test_dataset')
plt.savefig('figure/number_of_test_dataset.png',dpi=600)
plt.show()

# Bar_plot for validation data

Afridi=glob.glob('dataset/val/Afridi/*.*')
Mashrafe=glob.glob('dataset/val/Mashrafe/*.*')
Messi=glob.glob('dataset/val/Messi/*.*')
Ronaldo=glob.glob('dataset/val/Ronaldo/*.*')
Will_Smith=glob.glob('dataset/val/Will_Smith/*.*')

classnames = ['Afridi','Mashrafe','Messi','Ronaldo','Will_Smith']
Numbers = [len(Afridi),len(Mashrafe),len(Messi),len(Ronaldo),len(Will_Smith)]
plt.bar(classnames,Numbers,color=['green', 'red', 'tomato', 'blue', 'black'])
plt.title('Bar_Plot of Validation _dataset')
plt.savefig('figure/number_of_validation_dataset.png',dpi=600)
plt.show()