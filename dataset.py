
import splitfolders  

input_folder = 'all_images/'
def create_dataset():
    splitfolders.ratio(input_folder, output="dataset", 
                   seed=42, ratio=(.75, .15, .1), 
                   group_prefix=None)





