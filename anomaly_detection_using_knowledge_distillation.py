# -*- coding: utf-8 -*-

from google.colab import drive
drive.mount('/content/drive')

from google.colab import drive
drive.mount('/content/drive')
drive.mount("/content/drive", force_remount=True)

"""# New Section

# New Section
"""

!pip install keras-layer-normalization

class Config:
  DATASET_PATH ="/content/drive/MyDrive/UCSD_Anomaly/UCSDped1/Train/Train001"
  NEW_SET_PATH ="/content/drive/MyDrive/UCSD_Anomaly/UCSDped1/Train/Train002"
  SINGLE_TEST_PATH = "/content/drive/MyDrive/UCSD_Anomaly/UCSDped1/Train/Train001"
  BATCH_SIZE = 3
  EPOCHS = 3
  MODEL_PATH = "/content/drive/MyDrive/UCSD_Anomaly/model.hdf5"
  teacher_model_path = "/content/drive/MyDrive/UCSD_Anomaly/teacher.hdf5"
  student_model_path = "/content/drive/MyDrive/UCSD_Anomaly/student.hdf5"

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from os import listdir
import os
from os.path import isfile, join, isdir
from PIL import Image
import numpy as np
import shelve
def get_clips_by_stride(stride, frames_list, sequence_size):
    print("\nIn getclips()")
    clips = []
    sz = len(frames_list)
    clip = np.zeros(shape=(sequence_size, 256, 256, 1))
    cnt = 0
    for start in range(0, stride):
        for i in range(start, sz, stride):
            clip[cnt, :, :, 0] = frames_list[i]
            cnt = cnt + 1
            if cnt == sequence_size:
                clips.append(np.copy(clip))
                cnt = 0
    print("\nin get clips by stride\n",clips)
    return clips

def get_training_set():
    clips = []
    image_folder_path = Config.DATASET_PATH

    # List all files in the folder
    file_list = os.listdir(image_folder_path)

    # Filter out only image files (e.g., with ".tif" extension)
    image_files = [file for file in file_list if file.lower().endswith((".tif"))]

    all_frames = []
            # Loop over all the images in the folder (0.tif, 1.tif, ..., 199.tif)
    for c in image_files:
        print(c)
        image_path = os.path.join(image_folder_path, c)
      #Check for file extension
        img = Image.open(image_path).resize((256, 256))
        img = np.array(img, dtype=np.float32) / 256.0
        all_frames.append(img)
    print("printing",all_frames)
    # Get the 10-frames sequences from the list of images after applying data augmentation
    for stride in range(1, 3):
        clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))

    print("Number of clips:", len(clips))  # Print the number of clips
    return clips


def get_new_test_set():
    clips = []
    image_folder_path = Config.NEW_SET_PATH

    # List all files in the folder
    file_list = os.listdir(image_folder_path)

    # Filter out only image files (e.g., with ".tif" extension)
    image_files = [file for file in file_list if file.lower().endswith((".tif"))]

    all_frames = []
            # Loop over all the images in the folder (0.tif, 1.tif, ..., 199.tif)
    for c in image_files:
        print(c)
        image_path = os.path.join(image_folder_path, c)
      #Check for file extension
        img = Image.open(image_path).resize((256, 256))
        img = np.array(img, dtype=np.float32) / 256.0
        all_frames.append(img)
    print("printing",all_frames)
    # Get the 10-frames sequences from the list of images after applying data augmentation
    for stride in range(1, 3):
        clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))

    print("Number of clips:", len(clips))  # Print the number of clips
    return clips



'''
def get_training_set():

    clips = []
    # loop over the training folders (Train000,Train001,..)
    for f in sorted(listdir(Config.DATASET_PATH)):
        if isdir(join(Config.DATASET_PATH, f)):
            all_frames = []
            # loop over all the images in the folder (0.tif,1.tif,..,199.tif)
            for c in sorted(listdir(join(Config.DATASET_PATH, f))):
                if str(join(join(Config.DATASET_PATH, f), c))[-3:] == "tif":
                    img = Image.open(join(join(Config.DATASET_PATH, f), c)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            # get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
    print(clips)
    return clips

def get_training_set():
    print("\nin gettrainingset()")
    clips = []

    # Loop over the training folders (Train000, Train001, ...)

    for f in sorted(listdir(Config.DATASET_PATH)):
       # print("f=",f)
        if isdir(join(Config.DATASET_PATH, f)):
            print("in if1")
            all_frames = []

            # Loop over all the images in the folder (0.tif, 1.tif, ..., 199.tif)
            for c in sorted(listdir(join(Config.DATASET_PATH, f))):

                if c.endswith(".tif"):  # Check for file extension
                    print("in if2")
                    img = Image.open(join(join(Config.DATASET_PATH, f), c)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))
        else:
            all_frames = []
            for c in sorted(join(Config.DATASET_PATH, f)):

                if c.endswith(".tif"):  # Check for file extension
                    print("in if2")
                    img = Image.open(join(join(Config.DATASET_PATH, f), c)).resize((256, 256))
                    img = np.array(img, dtype=np.float32) / 256.0
                    all_frames.append(img)

            # Get the 10-frames sequences from the list of images after applying data augmentation
            for stride in range(1, 3):
                clips.extend(get_clips_by_stride(stride=stride, frames_list=all_frames, sequence_size=10))

    print("Number of clips:", len(clips))  # Print the number of clips
    return clips
'''

import tensorflow as tf
tf.config.run_functions_eagerly(True)

import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization
from keras.models import Sequential, load_model
def get_model(reload_model=True):
    print("\n\n--->IN get model()")
    if not reload_model:
        return load_model(Config.MODEL_PATH,custom_objects={'LayerNormalization': LayerNormalization})
    print("\ncalling get traning set()\n")
    training_set = get_training_set()
    print(training_set)
    print("\n\ngot\ngetting trainingset()\n")
    training_set = np.array(training_set)
    print(training_set)
    print("\ngot\nreshapinhh()\n\n")
    training_set = training_set.reshape(-1,10,256,256,1)
    print(training_set)
    print("\ndone\ncalling seq()\n")
    seq = Sequential()
    print("done\n")
    print("***")
    seq.add(TimeDistributed(Conv2D(128, (11, 11), strides=4, padding="same"), batch_input_shape=(None, 10, 256, 256, 1)))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    seq.add(TimeDistributed(Conv2D(64, (5, 5), strides=2, padding="same")))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    # # # # #
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    seq.add(ConvLSTM2D(32, (3, 3), padding="same", return_sequences=True))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    seq.add(ConvLSTM2D(64, (3, 3), padding="same", return_sequences=True))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    # # # # #
    seq.add(TimeDistributed(Conv2DTranspose(64, (5, 5), strides=2, padding="same")))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    seq.add(TimeDistributed(Conv2DTranspose(128, (11, 11), strides=4, padding="same")))
    print("***")
    seq.add(LayerNormalization())
    print("***")
    seq.add(TimeDistributed(Conv2D(1, (11, 11), activation="sigmoid", padding="same")))
    print("***")
    print("Summary")
    print(seq.summary())
    print("Summary done")
    print("***")
    print("loss")
    seq.compile(loss='mse', optimizer=keras.optimizers.Adam(learning_rate=1e-4, decay=1e-5, epsilon=1e-6))
    print("loss calculated")
    print("***")
    print("fitting")
    seq.fit(training_set, training_set,batch_size=Config.BATCH_SIZE, epochs=Config.EPOCHS, shuffle=False)
    print("fitted\n")
    print("savinggg")
    r = seq.save(Config.MODEL_PATH)
    print(r)
    print("saveddd")
    print("***")

    return seq

def get_single_test():
    #sz = 200
    sz = 200
    test = np.zeros(shape=(sz, 256, 256, 1))
    cnt = 0
    for f in sorted(listdir(Config.SINGLE_TEST_PATH)):
        if str(join(Config.SINGLE_TEST_PATH, f))[-3:] == "tif":
            img = Image.open(join(Config.SINGLE_TEST_PATH, f)).resize((256, 256))
            img = np.array(img, dtype=np.float32) / 256.0
            test[cnt, :, :, 0] = img
            cnt = cnt + 1
    return test

import matplotlib.pyplot as plt

def evaluate():
    print("\nIN evalute")
    model = get_model(True)
    print("\ncalling get model\n")
   # model = get_model(True)
    print("got model")
    print("\ncalling single test\n")
    test = get_single_test()
    print("DOne")
    print(test.shape)
    sz = test.shape[0] - 10 + 1
    sequences = np.zeros((sz, 10, 256, 256, 1))
    # apply the sliding window technique to get the sequences
    for i in range(0, sz):
        clip = np.zeros((10, 256, 256, 1))
        for j in range(0, 10):
            clip[j] = test[i + j, :, :, :]
        sequences[i] = clip

    print("got data")
    # get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(sequences,batch_size=4)
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(sequences[i],reconstructed_sequences[i])) for i in range(0,sz)])
    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()



tf.config.run_functions_eagerly(True)
tf.data.experimental.enable_debug_mode()

evaluate()

"""Distillation involves using the predictions of the teacher model as soft targets for training the student model. The goal is to have the student model learn from the teacher's knowledge and generalize its understanding to new data.

**create_teacher_model()**: defines a function that creates a teacher model using Convolutional Neural Network (CNN) layers. The model architecture consists of convolutional layers with max-pooling, a flatten layer, and fully connected layers.
The input shape is (256, 256, 1), representing grayscale images

**get_teacher_predictions()** takes the trained teacher model and a data sample as input. It reshapes the data sample to match the expected input shape of the teacher model and then uses the teacher model to make predictions on the reshaped data.

**create_student_model()** defines a function to create a student model with a similar architecture to the teacher model but with smaller filter sizes. The purpose of the student model is to learn from the teacher's predictions.

**train_student_with_distillation()** trains the student model using distillation. The function compiles the student model with a loss function that combines mean squared error and KL divergence losses.

It iterates through the training set and uses the teacher's predictions as targets to guide the student learning.

**get_distillation_loss()** estimates the distillation loss in training the student model.

"""

import keras
from keras.layers import Conv2DTranspose, ConvLSTM2D, BatchNormalization, TimeDistributed, Conv2D, LayerNormalization, Input, MaxPooling2D, Flatten, Dense
from keras.models import Sequential, load_model, Model
from keras.losses import KLDivergence, MeanSquaredError
import keras.backend as K
import numpy as np
import matplotlib.pyplot as plt



def create_teacher_model():
    input_shape = (256, 256, 1)

    inputs = Input(shape=input_shape)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(128, activation='relu')(x)
    outputs = Dense(256, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='teacher_model')
    return model

def get_teacher_predictions(teacher_model, data):
      # Convert the list of images to a NumPy array
        data_array = np.array(data)

        # Ensure that the input data matches the expected input shape of the teacher model
        # For example, if the teacher model expects an input shape of (256, 256, 1), you can reshape the data as follows:
        reshaped_data = data_array.reshape(-1,256, 256, 1)

        predict = teacher_model.predict(reshaped_data)
        print(predict)
        return predict


def create_student_model():
    input_shape = (256, 256, 1)

    inputs = Input(shape=input_shape)
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(inputs)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D((2, 2), padding='same')(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    outputs = Dense(256, activation='sigmoid')(x)

    model = Model(inputs, outputs, name='student_model')
    return model

def train_student_with_distillation(teacher_model, student_model, alpha, training_set, teacher_predictions):
    # Convert lists to NumPy arrays
    training_set = np.array(training_set)
    teacher_predictions = np.array(teacher_predictions)

    # Train the student model using teacher's predictions as targets
    student_model.compile(loss=get_distillation_loss(alpha, teacher_predictions), optimizer='adam')

    loss_history = []  # List to store loss values

    for i in range(len(training_set)):
        sample = training_set[i]
        target = teacher_predictions[i]

        sample = np.reshape(sample, (10, 256, 256, 1))
        target = np.reshape(target, (10, 256))

        # Fit the student model with the current sample and target
        history = student_model.fit(sample, target, batch_size=1, epochs=Config.EPOCHS, verbose=0)
        loss_history.extend(history.history['loss'])

    return loss_history



'''
#def train_student_with_distillation(teacher_model, student_model, alpha, training_set, validation_set):
def train_student_with_distillation(teacher_model, student_model, alpha, training_set, teacher_predictions):
    # Convert lists to NumPy arrays
    training_set = np.array(training_set)
    teacher_predictions = np.array(teacher_predictions)

    # Train the student model using teacher's predictions as targets
    student_model.compile(loss=get_distillation_loss(alpha, teacher_predictions), optimizer='adam')

    loss_history = []  # List to store loss values


    for i in range(len(training_set)):
        sample = training_set[i]
        target = teacher_predictions[i]

        sample = np.reshape(sample, (10, 256, 256, 1))
        target = np.reshape(target, (10, 256))

        history = student_model.fit(sample, target, batch_size=1, epochs=Config.EPOCHS, verbose=0)
        loss_history.append(history.history['loss'])


        student_model.fit(sample, target, batch_size=1, epochs=Config.EPOCHS)

        return loss_history
'''
def get_distillation_loss(alpha, teacher_predictions):

    def distillation_loss(y_true, y_pred):
        # Mean squared error between student predictions and ground truth
        mse_loss = K.mean(K.square(y_true - y_pred), axis=-1)

        # KL Divergence between teacher predictions and student predictions
        kl_loss = K.mean(K.square(teacher_predictions - y_pred), axis=-1)

        # Combine the two losses using the distillation parameter alpha
        total_loss = alpha * kl_loss + (1 - alpha) * mse_loss
        return total_loss

    return distillation_loss







def main():

    teacher_model = create_teacher_model()
    student_model = create_student_model()
    data_set = get_training_set()

    train_ratio = 0.5  # Example ratio, adjust as needed
    num_samples = len(data_set)
    num_train_samples = int(train_ratio * num_samples)
    training_set = data_set[:num_train_samples]
    validation_set = data_set[num_train_samples:]
    alpha = 0.5

    # Generate teacher predictions for the training set
   # teacher_predictions = teacher_model.predict(np.array(training_set))
   # Generate teacher predictions for each sample in the validation set
    teacher_predictions = []
    for sample in validation_set:
        teacher_prediction = get_teacher_predictions(teacher_model, sample)
        teacher_predictions.append(teacher_prediction)

   # train_student_with_distillation(teacher_model, student_model, alpha, training_set, teacher_predictions)

    loss_history = train_student_with_distillation(teacher_model, student_model, alpha, training_set, teacher_predictions)

    teacher_model.save(Config.teacher_model_path)
    student_model.save(Config.student_model_path)

    print("\n\nPrinting loss list:\n")
    print(loss_history)
    # Plot the loss graph
    plt.plot(loss_history)
    plt.title('Training Loss')
    plt.xlabel('Training Iteration')
    plt.ylabel('Loss')
    plt.show()


if __name__ == "__main__":


      main()

"""1. Loading the student model which is trained already
2. Loading the test data.
3. Loading test data.
4. creating sequences and calculatig regularity score which says well it identitifes correctly
5. A low score says that it does well. It's predicting good.


**Graph**

When the graph is high, it means the student model is having trouble understanding the patterns. This could indicate that something unusual or unexpected is happening in the data.

When the graph is low, it means the student model is doing a good job at recognizing the patterns. This suggests that the data is following the expected patterns.

Need to compare the values to a threshold value
----> yet to decide the threshold. Will do once run on GPU



"""

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model
from keras.utils import custom_object_scope

def evaluate_student():
    print("\nIN evaluate_student")

    # Load the trained student model
    with custom_object_scope({'distillation_loss': get_distillation_loss}):
        model = load_model(Config.student_model_path)

    print("\nLoaded trained student model")

    # Load test data or generate your own test data
    test_data = get_new_test_set()  # Replace with your method to load or generate test data
    test_data_array = np.array(test_data)

    # Get the shape of the test data array
    num_samples, seq_length, height, width, channels = test_data_array.shape

    # Initialize an array to store the sequences
    sequences = np.zeros((num_samples * (seq_length - 10 + 1), 10, height, width, channels))

    # Apply the sliding window technique to get the sequences
    for i in range(0, num_samples):
        for j in range(0, seq_length - 10 + 1):
            clip = test_data_array[i, j:j+10, :, :, :]
            sequences[i * (seq_length - 10 + 1) + j] = clip

    print("Got data")

    # Reshape sequences to match the model's input shape
    reshaped_sequences = sequences.reshape(-1, 256, 256, 1)

    # Get the reconstruction cost of all the sequences
    reconstructed_sequences = model.predict(reshaped_sequences, batch_size=4)

    # Calculate the reconstruction cost for each sequence
    sequences_reconstruction_cost = np.array([np.linalg.norm(np.subtract(reshaped_sequences[i], reconstructed_sequences[i])) for i in range(0, reshaped_sequences.shape[0])])

    sa = (sequences_reconstruction_cost - np.min(sequences_reconstruction_cost)) / np.max(sequences_reconstruction_cost)
    sr = 1.0 - sa

    # Plot the regularity scores
    plt.plot(sr)
    plt.ylabel('regularity score Sr(t)')
    plt.xlabel('frame t')
    plt.show()

# Call the evaluation function for the student model
evaluate_student()