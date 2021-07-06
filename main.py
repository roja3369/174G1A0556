
import os
import pathlib
import PIL
import PIL.Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
from glob import glob
from tqdm import tqdm
tqdm().pandas()
from sklearn import preprocessing
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score, matthews_corrcoef, roc_auc_score, confusion_matrix
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Model
# Functions to extract the true, false positive and true false negative
def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]

def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]

def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]

def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]

def prepare_dataset(path,label):
    x_train=[]
    y_train=[]
    all_images_path=glob(path+'/*.jpg')
    for img_path in tqdm(all_images_path) :
            img=load_img(img_path, target_size=(150,150))
            img=img_to_array(img)
            img=img/255.0
            x_train.append(img)
            y_train.append(label)
    return x_train,y_train
    

def plot_learning_curve(history, name):
    '''
    Function to plot the accuracy curve
    @param history: (history object) containing all the relevant information about the training
    @param name: (string) name of the model
    '''
    # extract informations 
    acc     = history.history["accuracy"]
    val_acc = history.history["val_accuracy"]
    loss    = history.history["loss"] 
    val_loss= history.history["val_loss"] 
    epochs  = range(1, len(acc)+1)
    # plot accuracy
    plt.plot(epochs, acc, 'b-o', label='Training loss')
    plt.plot(epochs, val_acc, 'g', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.savefig(name+"_accuracy.png") # save the picture, put into comment if you don't want
    plt.figure()
    # plot losses
    plt.plot(epochs, loss, 'b-o', label='Training loss')
    plt.plot(epochs, val_loss, 'g', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.savefig(name+"_losses.png") # save the picture, put into comment if you don't want
    plt.show()

# creating paths to retrieve the data
ROOT = r"C:\Users\roj\Desktop\Main\PROJECTS\hyp_spec_cv_da\hyp_spec_cv_da"
FOLDER = "images"

TRAIN = os.path.join(ROOT, FOLDER, 'seg_train/seg_train')
TEST = os.path.join(ROOT, FOLDER, 'seg_test/seg_test')

data_dir = pathlib.Path(TRAIN)
test_dir = pathlib.Path(TEST)

#print(data_dir)
image_count = len(list(data_dir.glob('*/*.jpg')))

print(f"The number of images contained in the train set is {image_count}")

#to see img
#cube = list(data_dir.glob('cube/*'))
#PIL.Image.open(str(cube[0]))

# Train dataset
labels = os.listdir(TRAIN)
x = []
y = []
for label in labels:
    x_, y_ = prepare_dataset(os.path.join(TRAIN, label), label)
    x.extend(x_)
    y.extend(y_)
x = np.array(x)
y = np.array(y)

labels = os.listdir(TEST)
x_test = []
y_test = []
for label in labels:
    x_, y_ = prepare_dataset(os.path.join(TEST, label), label)
    x_test.extend(x_)
    y_test.extend(y_)
x_test = np.array(x_test)
y_test = np.array(y_test)

# create a validation set 
from sklearn.model_selection import train_test_split
train_x, valid_x, y_train, y_valid = train_test_split(x, y, random_state=42, stratify=y, test_size=0.2)
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(y_train)
valid_y = encoder.transform(y_valid)
test_y  = encoder.transform(y_test)
print(len(train_x), len(valid_x), len(train_y), len(valid_y ))

# initialize values to store the results 
dict_hist = {}
df_results = pd.DataFrame()
num_classes = 2
#Creating Model
model = tf.keras.Sequential([
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.5),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Dropout(0.3),
  layers.Conv2D(32, 3, activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])
#compiling Model
model.compile(
  optimizer='adam',
  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(patience=3)
history = model.fit(
  train_x,train_y,
  validation_data=(valid_x, valid_y), callbacks=[es], batch_size=5,
   epochs=40
)


dict_hist["from_scratch"] = history

plot_learning_curve(history, "image_classif_from_scratch")
#DATA AUGMENTATION
from tensorflow.keras.preprocessing.image import ImageDataGenerator
data_gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')
        
#Making Train and Test Dataset  
test_gen = ImageDataGenerator(rescale=1./255)
train_data_gen = data_gen.flow_from_directory(
                                        
                                        batch_size=5,
                                        directory=TRAIN,
                                        shuffle=True,
                                        target_size=(150, 150),
                                        class_mode='categorical'
                                        )
                                        
                                        
test_data_gen = test_gen.flow_from_directory(
                                        batch_size=1,
                                        directory=TEST,
                                        shuffle=True,
                                        target_size=(150, 150),
                                        class_mode='categorical'
                                        )

num_classes = 2
#Creating a Model
model = tf.keras.Sequential([
    #data_augmentation,
  layers.Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)),
  layers.MaxPooling2D(2,2),
  layers.Dropout(0.5),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Dropout(0.3),
  layers.Conv2D(128, (3,3), activation='relu'),
  layers.MaxPooling2D(2,2),
  layers.Flatten(),
  layers.Dropout(0.5),  
  layers.Dense(512, activation='relu'),
  layers.Dense(2, activation='softmax')
])

#Compiling a Model
model.compile(
  optimizer='adam',
  loss= 'categorical_crossentropy',
  metrics=['accuracy'])
es = tf.keras.callbacks.EarlyStopping(patience=3)
history2 = model.fit(
  train_data_gen,
    steps_per_epoch=10//2,
  validation_data=test_data_gen, callbacks=[es],
   epochs=40
)


plot_learning_curve(history2, "image_classif_data_augmentation")


























