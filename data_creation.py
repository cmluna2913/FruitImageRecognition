# import libraries to turn our images in a useable form
import numpy as np
import os
import cv2

# this will be to package our data so we can use it in our other notebook
import pickle
import random

def data_creation(img_size, flag, categories, data_dir):
    '''
    Inputs: (img_size, flag, categories, data_dir)
    
    Outputs: (X,y)
        returns (X, y) where X and y are sets of data.
        X is an array of data in the shape of (n, img_size, img_size, 1)
        where n is the number of elements.
        y is an array of labels in the shape of (n,)
        
    img_size:
        Size of wanted image size as an integer.
    flag:
        Flag input for cv2.imread() as an integer.
        1 to load an image in color.
        0 to load an image in grayscale.
        -1 to load an image unchanged. This function didn't take into
        account -1, so it is unknown how it will work exactly.
    categories:
        List of directory names to load images.
        E.g., categories = ['Apple', 'Banana', 'Carambola']
        as these are the names given for the folders containing
        the images.
    data_dir:
        Directory path that contains the data.
        E.g., 'C:/User/Username/Desktop/Images'
        where '/Images' contains the images split
        into their own, respective directory for each category.
    '''
    data = []
    # Iterating through each category and getting to the images
    for category in categories:
        path = os.path.join(data_dir, category)
        cat_num = categories.index(category)
        # Depending on our flag, we import the images as grayscale or color and resize.
        # [resize_img_arr, cat_num] adds the array and label to our data list
        if flag==0:
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), flag)
                resize_img_arr = cv2.resize(img_arr, (img_size, img_size))
                data.append([resize_img_arr, cat_num])
        elif flag==1:
            for img in os.listdir(path):
                img_arr = cv2.imread(os.path.join(path, img), flag)
                img_rgb = cv2.cvtColor(img_arr, cv2.COLOR_BGR2RGB)
                resize_img_arr = cv2.resize(img_rgb, (img_size, img_size))
                data.append([resize_img_arr, cat_num])
    #Shuffle our data around so it randomized
    random.shuffle(data)
    
    #Create our X and y sets from our data list
    X = []
    y = []
    for arr, label in data:
        X.append(arr)
        y.append(label)
        
    # Reshaping our arrays to (img_size, img_size, channels)
    if flag==0:
        X = np.array(X).reshape(-1, img_size, img_size, 1)
    elif flag==1:
        X = np.array(X).reshape(-1, img_size, img_size, 3, 1)
    y = np.array(y)
    return X, y


def pickle_me(X, y, X_name, y_name):
    '''
    Inputs: (X, y, , file_names)
    
    Outputs:
        Pickles data for future use.
        
    X:
        X data
    y:
        y data
    filie_names:
        Selected name for our X and y data, respectively, as a list of strings.
        ".pickle" will be addad at the end of each name.
        E.g., ["X_data", "y_data"] will be saved as "X_data.pickle", "y_data.pickle"
    '''
    #Pickles our data
    X_pickle_out = open(X_name+'.pickle', 'wb')
    pickle.dump(X, X_pickle_out)
    X_pickle_out.close()
        
    y_pickle_out = open(y_name+'.pickle', 'wb')
    pickle.dump(y, y_pickle_out)
    y_pickle_out.close()
    
    
    
def get_pickle(X_file, y_file):
    '''
    Inputs: (X_file, y_file)
    
    Outputs: (X,y)
        Return X and y data from a pickle file.
        
    X_file:
        File name for X data as a string.
    y_file:
        File name for y data as a string.
    '''
    #Opens up our pickled data
    pickle_X = open(X_file, 'rb')
    X = pickle.load(pickle_X)
    
    pickle_y = open(y_file, 'rb')
    y = np.array(pickle.load(pickle_y))
    return X,y