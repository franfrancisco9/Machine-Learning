
import numpy as np
import sklearn
# Neural Networks
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.callbacks import EarlyStopping
# To deal with Imbalanced dataset
from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import NearMiss
from tensorflow.keras.layers import RandomRotation, RandomFlip
# Plot
import matplotlib.pyplot as plt
# Sklearn/Evaluation
from sklearn.metrics import classification_report, confusion_matrix, f1_score, ConfusionMatrixDisplay, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold

# -------------------------- 1 - Checkout the Data --------------------------- #
x_train = np.load("Xtrain_Classification1.npy")
y_train = np.load("ytrain_Classification1.npy")
x_test = np.load("Xtest_Classification1.npy")

print("Training Images:", x_train.shape)
print("Images Class:", y_train.shape)
print("Testing Images:", x_test.shape)

x_train_reshaped = x_train.reshape(x_train.shape[0], 28, 28, 3)
nevus = 0
melanoma = 0

for label in y_train:
  if(label==0):
    nevus+=1
  if(label==1):
    melanoma+=1

print('Total number of nevus:', nevus)
print('Total number of melanoma', melanoma)

# Display the first five images in training data
# for i in range(5):
#   plt.imshow(x_train_reshaped[i])
#   if(y_train[i]==0):
#     plt.title("Ground Truth : Spot")
#   if(y_train[i]==1):
#     plt.title("Ground Truth : Eyespot")
#   plt.show()

# Find the train labels
classes = np.unique(y_train)
classes_num = len(classes)
print('Total number of outputs : ', classes_num)
print('Output classes : ', classes)

# --------------------------- 2 - Process the data --------------------------- #

# Standardizing the data between 0 and 1
# Divide RGB channels by 255 to make neural network inputs small.
x_train = x_train.astype('float32')
x_train = x_train/255

# REDUZIR NÃšMERO DE EPOCHS

# Split data into training and validation
# Use stratify to keep proportion of classes in training and testing
x_train_new, x_val, y_train_new, y_val = train_test_split(x_train, y_train, test_size = 0.2, stratify=y_train)
print('Training data shape : ', x_train_new.shape, y_train_new.shape)
print('Validation data shape : ', x_val.shape, y_val.shape)

# --------------- 3 - Deal with Imbalance in Training Data ------------------- #

# Different Option:

# 1 - Oversampling class 1 with SMOTE since it has fewer training examples.
# 2 - Oversampling class 1 using data from class 1 itself. (RandomOverSampler)
# 3 - Give classes indices diffent weights for the loss function
# (during training only). This can be useful to tell the model to "pay more
# attention" to samples from an under-represented class.
# 4 - UnderSample Majority class

train_nevus = 0
train_melanoma = 0 # (Undersampled class)
training_examples = 0

for label in y_train_new:
  if(label==0):
    train_nevus+=1
  if(label==1):
    train_melanoma+=1
  training_examples+=1

print('Total number of nevus for training:', train_nevus)
print('Total number of melanoma for training:', train_melanoma)
print('Total number of training examples:', training_examples)

# Option 1
# sm = SMOTE(random_state=42)
# x_train_new, y_train_new = sm.fit_resample(x_train_new, y_train_new)

# Option 2
ros = RandomOverSampler(random_state=42)
x_train_new, y_train_new = ros.fit_resample(x_train_new, y_train_new)

# Option 3
# weight_0=1
# weight_1=train_nevus/train_melanoma
# print("Weight of undersampled class:", weight_1)
# class_weight = {0: weight_0,1: weight_1,}

# Option 4
# nm = NearMiss()
# x_train_new, y_train_new = nm.fit_resample(x_train_new, y_train_new)

print("New training images shape:", x_train_new.shape)
print("New training labels shape:", y_train_new.shape)

# Label Convertion to one hot enconding - Output is a vector of 2 dimentions
y_train_new = keras.utils.to_categorical(y_train_new)
y_val = keras.utils.to_categorical(y_val)

print("New training labels shape after one hot encoding:", y_train_new.shape)

# ---------------------------------------------------------------------------- #
# ------------------------------------ MLP ----------------------------------- #
# ---------------------------------------------------------------------------- #

# ------------------------- 1 - Create the Network --------------------------- #

# Create sequential model
# Sequential: Keras model that adds up layers
# Dense: Layer where all input connect to each neuron (totally connected)
# Dropout: Layer used during training that randomly eliminates some connections to avoid overfitting
model = Sequential()

# Convert each image matrix (30Ã—30x3=2700) to an array of dimension 2700
# which will be fed to the network as a single feature.
# Add Input Layer
model.add(InputLayer(input_shape=(28*28*3,)))

# First Hiden Layer with 128 neurons
model.add(Dense(128, activation = 'relu'))

# 20% Dropout
model.add(Dropout(0.2))

# Second Hiden Layer with 64 neurons
model.add(Dense(64, activation = 'relu'))

# 20% Dropout
model.add(Dropout(0.2))

# Third Hiden Layer with 32 neurons
model.add(Dense(32, activation = 'relu'))

# 20% Dropout
model.add(Dropout(0.2))

# Final classification Layer (Softmax layer)
# 1 neuron for each output class. Softmax divides the probability of each class.
model.add(Dense(2, activation = 'softmax'))

# Check if the model is working
model.summary()

# --------------------------- 2 - Configure the Network ---------------------- #

# Fit MLP model
model.compile(loss='binary_crossentropy', optimizer='Adam', metrics=['accuracy'])

# Stopping Monitor - USE EARLY STOPPING TO SUBMIT !!
callback = EarlyStopping(patience = 10, restore_best_weights = True)

# ------------------- 3 - Train and Evaluate the Model ----------------------- #

# Train using some training data sample (Mini-Batch Mode)
batch_size = 128
epochs = 20
history = model.fit(x_train_new, y_train_new, batch_size=batch_size, callbacks = [callback],epochs=epochs,verbose=1, validation_data=(x_val, y_val))

# # Watch training results - LOSS
# plt.plot(history.history['loss'], color='b', label="Training loss")
# plt.plot(history.history['val_loss'],color='r', label="validation loss")
# plt.title("Model Loss Plot (Without Early Stopping) - MLP")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(loc='best', shadow=True)
# plt.show()

# # Watch training results - ACCURACY
# plt.plot(history.history['accuracy'], color='b', label="Training accuracy")
# plt.plot(history.history['val_accuracy'],color='r', label="Validation accuracy")
# plt.title("Model Accuracy Plot (With Early Stopping) - MLP")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(loc='best', shadow=True)
# plt.show()

# Test
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Test
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict Class for all validation data
predict_x = model.predict(x_val)
y_pred =  np.argmax(predict_x,axis=1)

# Turn one-hot enconding back to class format
y_test_c = np.argmax(y_val, axis=1)
target_names = ['0', '1']

print('--------------- Classification Report for MLP -----------------')
print(classification_report(y_test_c, y_pred, target_names=target_names))
print("Balanced Accuracy",balanced_accuracy_score(y_test_c, y_pred))
cm = confusion_matrix(y_test_c, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix.png')

# ---------------------------------------------------------------------------- #
# ----------------------------------- CNN ------------------------------------ #
# ---------------------------------------------------------------------------- #

# CNN Basic Structure goes as follows:
# - Convolution -> Pooling -> Convolution -> Pooling -> Fully Connected Layer -> Output.
# - Conv takes a 3x3 filter slides it every pixel (strides = (1,1)) whilst calculating
# the dot product of the filter and the current position in x_train_reshaped.
# - Pooling takes a 2x2 window "peak" at x_train_reshaped and chooses the highest value
# (MaxPooling) inside such window while moving at a 1x1 rate (strides)
# - Every Convolution step + Pooling step is a Hidden Layer
# - The first layer is the input layer and the last layer is the output layer. Everything in between are the hidden layers.
# - Problem - How many Hidden Layers do we need?
# - Problem - How many epochs do we need? ----> Use Early Stopping just like in MLP
# - Careful with overfitting!!

model2 = Sequential()
print(x_train_reshaped.shape)
# Ensure the input shape is correct (height, width, channels)
input_shape = (x_train_reshaped.shape[1], x_train_reshaped.shape[2], x_train_reshaped.shape[3])

# Input Layer
model2.add(InputLayer(input_shape=input_shape))

# Data Augmentation Layers - Add the preprocessing/augmentation layers.
model2.add(RandomFlip("horizontal_and_vertical"))
model2.add(RandomRotation(0.2))

# First Hidden Layer - After each Conv2D operation the CNN applies a ReLU to the
# feature map introducing nonlinearity to the model
model2.add(Conv2D(32, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))

# 20% Dropout
model2.add(Dropout(0.2))

# Second Hidden Layer
model2.add(Conv2D(64, (3, 3), activation = 'relu'))
model2.add(MaxPooling2D(pool_size = (2, 2)))

# 20% Dropout
model2.add(Dropout(0.2))

# Third Hidden Layer
model2.add(Conv2D(128, (3, 3), activation='relu'))

# 20% Dropout
model2.add(Dropout(0.2))

# Fourth Hidden (Fully connected) Layer
model2.add(Flatten())  # This converts our 3D feature maps to 1D feature vectors for the fully connected layer
model2.add(Dense(64))  # 64 nodes

# Output Layer
model2.add(Dense(2, activation = 'softmax')) # softmax converts a vector of values to a probability distribution

# Check if the model is working
model2.summary()

# binary_crossentropy since data can only be eyespot or spot
# adam stands for adaptive momentum estimation:
# - Combination of Momentum and RMSP algorithms
# - Momentum algorithm is used to accelerate the gradient descent algo by taking in consideration exponentially weighted avg of gradients (converges to minima faster)
# - RMSP stands for root mean square propagation and takes the exponential moving average of squared gradients
# - It is basically a more optmized gradient descent
model2.compile(loss ='binary_crossentropy',
              optimizer = 'adam',
              metrics = ['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])

# Reshape training data to input format of CNN (Which is different from the input of the MLP)
x_train_CNN = x_train_new.reshape(x_train_new.shape[0], 28, 28, 3)
x_val_CNN = x_val.reshape(x_val.shape[0], 28, 28, 3)
batch_size_CNN = 128
epochs_CNN = 100
history = model2.fit(x_train_CNN, y_train_new, batch_size=batch_size_CNN, epochs=epochs_CNN,callbacks = [callback],verbose=1, validation_data=(x_val_CNN, y_val))

# # Watch training results - LOSS
# plt.plot(history.history['loss'], color='b', label="Training loss")
# plt.plot(history.history['val_loss'],color='r', label="validation loss")
# plt.title("Model Loss Plot (Without Early Stopping) - CNN")
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.legend(loc='best', shadow=True)
# plt.show()

# # Watch training results - ACCURACY
# plt.plot(history.history['accuracy'], color='b', label="Training accuracy")
# plt.plot(history.history['val_accuracy'],color='r', label="Validation accuracy")
# plt.title("Model Accuracy Plot (With Early Stopping) - CNN")
# plt.xlabel("Epochs")
# plt.ylabel("Accuracy")
# plt.legend(loc='best', shadow=True)
# plt.show()

# Test
score = model.evaluate(x_val, y_val, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Predict Class for all validation data
predict_x = model2.predict(x_val_CNN)
y_pred =  np.argmax(predict_x,axis=1)

# Turn one-hot enconding back to class format
y_test_c = np.argmax(y_val, axis=1)
target_names = ['0', '1']

print('--------------- Classification Report for CNN -----------------')
print(classification_report(y_test_c, y_pred, target_names=target_names))

score = model2.evaluate(x_train_CNN, y_train_new)

print("Loss is - ", score[0], "\nAccuracy is - ", score[1], "\nPrecision is - ", score[2], "\nRecall is - ", score[3])
f1_score_a = 2 * ((score[2] * score[3])/(score[2] + score[3]))
print("\nAnalytical F1 SCORE IS -", f1_score_a)

f1_score_b = f1_score(y_test_c, y_pred, average = 'binary', pos_label = 1)
print("\nSklearn F1 SCORE IS -", f1_score_b)
print("Balanced Accuracy",balanced_accuracy_score(y_test_c, y_pred))
cm = confusion_matrix(y_test_c, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix_cn.png')

# -------------- FOR SUBMISSION - NOW USE ALL TRAINING DATA ------------------ #

# Reshape testing data to input format of CNN (Which is different from the input of the MLP)
x_test_CNN = x_test.reshape(x_test.shape[0], 28, 28, 3)
#Prediction for new data
y_submit = model2.predict(x_test_CNN, batch_size = batch_size_CNN, callbacks = [callback])
print("Shape of prediction:", y_submit.shape)

# Back to class format - for submission:
y_submit = np.argmax(y_submit, axis=1)
print("Shape of prediction for submission:", y_submit.shape)
np.save('output', y_submit)

# ---------------------------------------------------------------------------- #
# --------------------------------------- KNN -------------------------------- #
# ---------------------------------------------------------------------------- #

# The k-Nearest Neighbor classifier is by far the most simple machine learning
# and image classification algorithm. In fact, itâ€™s so simple that it doesnâ€™t
# actually â€œlearnâ€ anything. Instead, this algorithm directly relies on the
# distance between feature vectors (which in our case, are the raw RGB pixel
# intensities of the images). Simply put, the k-NN algorithm classifies unknown
# data points by finding the most common class among the k closest examples.
# Each data point in the k closest data points casts a vote, and the category
# with the highest number of votes wins.

# There are two clear hyperparameters that we are concerned with when running
# the k-NN algorithm. The first is obvious: the value of k. What is the optimal
# value of k? If itâ€™s too small (e.g., k = 1), then we gain efficiency but
# become susceptible to noise and outlier data points. However, if k is too
# large, then we are at risk of over-smoothing our classification results and
# increasing bias.
# The second parameter we should consider is the actual distance metric.
# Is the Euclidean distance the best choice? What about the Manhattan distance?

# train and evaluate a k-NN classifier on the raw pixel intensities
print("[INFO] evaluating k-NN classifier...")
# f1_score_b_max = 0
# f1_score_b = 0
# Kfolds = 5
# kf = KFold(n_splits=Kfolds, random_state=42, shuffle=True)
# for i in range(1, 100):
#   for train_new_index, test_validation_index in kf.split(x_train):
#     x_train_new_cv, x_val_cv = x_train[train_new_index], x_train[test_validation_index]
#     y_train_new_cv, y_val_cv = y_train[train_new_index], y_train[test_validation_index]
#     model = KNeighborsClassifier(i)
#     model.fit(x_train_new_cv, y_train_new_cv)
#     y_pred_knn = model.predict(x_val_cv)

#     f1_score_b += f1_score(y_val_cv, y_pred_knn)

#   f1_score_b = f1_score_b/Kfolds
#   if(f1_score_b > f1_score_b_max):
#     f1_score_b_max = f1_score_b
#     n_neighbors = i
#   f1_score_b = 0
# for i in range(1, 100):
#   model = KNeighborsClassifier(i)
#   model.fit(x_train_new, y_train_new)
#   y_pred_knn = model.predict(x_val)
#   y_pred_knn =  np.argmax(y_pred_knn,axis=1)

#   # Turn one-hot enconding back to class format
#   y_test_c = np.argmax(y_val, axis=1)
#   target_names = ['0', '1']

#   print(classification_report(y_test_c, y_pred_knn, target_names=target_names))

#   f1_score_b = f1_score(y_test_c, y_pred_knn, average = 'binary', pos_label = 1)
#   print("F1_SCORE for k =", i, "is -", f1_score_b)

#   if(f1_score_b > f1_score_b_max):
#     f1_score_b_max = f1_score_b
#     n_neighbors = i

  # Display Confusion Matrix
  # cm = confusion_matrix(y_test_c, y_pred_knn)
  # disp = ConfusionMatrixDisplay(confusion_matrix=cm)
  # disp.plot()
  # plt.show()
model = KNeighborsClassifier(1)
model.fit(x_train_new, y_train_new)
y_pred_knn = model.predict(x_val)

y_pred_knn =  np.argmax(y_pred_knn,axis=1)

# Turn one-hot enconding back to class format
y_test_c = np.argmax(y_val, axis=1)
target_names = ['0', '1']

print(classification_report(y_test_c, y_pred_knn, target_names=target_names))
# Display Confusion Matrix
cm = confusion_matrix(y_test_c, y_pred_knn)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
print("Balanced Accuracy",balanced_accuracy_score(y_test_c, y_pred_knn))
plt.show()
f1_score_b = f1_score(y_test_c, y_pred_knn, average = 'binary', pos_label = 1)
print("\nSklearn F1 SCORE IS -", f1_score_b)


# ---------------------------------------------------------------------------- #
# ------------------------------- Naive Bayes -------------------------------- #
# ---------------------------------------------------------------------------- #

# Called naive sincee we assume the features r independent of each other
# In Gaussian Naive Bayes, continuous values associated with each feature are assumed to be distributed according to a Gaussian distribution
#
# Turn one-hot enconding back to class format
# model.fit of GNB does not accept 2-D y_train array

y_train_nb = np.argmax(y_train_new, axis=1)
y_true_nb = np.argmax(y_val, axis = 1)
target_names = ['0', '1']
print("[INFO] evaluating Naives Bayes classifier...")
model = GaussianNB()
model.fit(x_train_new, y_train_nb)
y_pred_nb = model.predict(x_val)
print(classification_report(y_true_nb, y_pred_nb, target_names=target_names))
cm = confusion_matrix(y_true_nb, y_pred_nb)
print("Balanced Accuracy",balanced_accuracy_score(y_true_nb, y_pred_nb))
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.savefig('confusion_matrix_nb.png')
