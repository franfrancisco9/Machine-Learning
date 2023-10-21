
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

# Parameter values were obtained with loops in another file

# ============================== Balanced Accuracy =============================== #
class BalancedAccuracy(tf.keras.metrics.Metric):
    def __init__(self, name="balanced_accuracy", **kwargs):
        super(BalancedAccuracy, self).__init__(name=name, **kwargs)
        
        # Initialize the confusion matrix counters
        self.tp = self.add_weight(name="tp", initializer="zeros")  # True positives
        self.fp = self.add_weight(name="fp", initializer="zeros")  # False positives
        self.tn = self.add_weight(name="tn", initializer="zeros")  # True negatives
        self.fn = self.add_weight(name="fn", initializer="zeros")  # False negatives

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred_class = tf.argmax(y_pred, axis=1)
        y_true_class = tf.argmax(y_true, axis=1)

        # Calculate the confusion matrix components
        self.tp.assign_add(tf.reduce_sum(tf.cast((y_true_class == 1) & (y_pred_class == 1), tf.float32)))
        self.fp.assign_add(tf.reduce_sum(tf.cast((y_true_class == 0) & (y_pred_class == 1), tf.float32)))
        self.tn.assign_add(tf.reduce_sum(tf.cast((y_true_class == 0) & (y_pred_class == 0), tf.float32)))
        self.fn.assign_add(tf.reduce_sum(tf.cast((y_true_class == 1) & (y_pred_class == 0), tf.float32)))

    def result(self):
        sensitivity = self.tp / (self.tp + self.fn + 1e-7)  # Adding epsilon to avoid division by zero
        specificity = self.tn / (self.tn + self.fp + 1e-7)
        
        return (sensitivity + specificity) / 2.0

    def reset_state(self):
        self.tp.assign(0.0)
        self.fp.assign(0.0)
        self.tn.assign(0.0)
        self.fn.assign(0.0)
# ============================== Load Data =============================== #
x_train = np.load("Xtrain_Classification1.npy")
y_train = np.load("ytrain_Classification1.npy")
x_test = np.load("Xtest_Classification1.npy")

print("Train size", x_train.shape)
print("Labels Size", y_train.shape)
print("Test Size", x_test.shape)

x_train_reshaped = x_train.reshape(x_train.shape[0], 28, 28, 3)

# Count the number of images for each class
nevus = np.count_nonzero(y_train == 0)
melanoma = np.count_nonzero(y_train == 1)
print('Total number of nevus:', nevus)
print('Total number of melanoma', melanoma)

# Find the train labels
classes = np.unique(y_train)
classes_num = len(classes)
print('Number of labels', classes_num)
print('Labels', classes)

# ============================== Preprocessing =========================== #

# Standardizing the data between 0 and 1
# Divide RGB channels by 255 to make neural network inputs small.
x_train = x_train.astype('float32')
x_train = x_train/255

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

# # Option 1
# sm = SMOTE(random_state=95789)
# x_train_new, y_train_new = sm.fit_resample(x_train_new, y_train_new)

# Option 2
ros = RandomOverSampler(random_state=95789)
x_train_new, y_train_new = ros.fit_resample(x_train_new, y_train_new)

# Option 3
# train_melanoma = np.count_nonzero(y_train_new == 1)
# train_nevus = np.count_nonzero(y_train_new == 0)
# weight_0=1
# weight_1=train_nevus/train_melanoma
# print("Weight of undersampled class:", weight_1)
# class_weight = {0: weight_0,1: weight_1,}



print("New training images shape:", x_train_new.shape)
print("New training labels shape:", y_train_new.shape)

# Label Convertion to one hot enconding - Output is a vector of 2 dimentions
y_train_new = keras.utils.to_categorical(y_train_new)
y_val = keras.utils.to_categorical(y_val)

print("New training labels shape after one hot encoding:", y_train_new.shape)

# ============================== MLP =========================== #
print('\n============================== MLP ===========================\n')
model = Sequential()
model.add(InputLayer(input_shape=(28*28*3,)))
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(32,activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2, activation='softmax'))

model.summary()

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[BalancedAccuracy()])

# Early stopping
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10, restore_best_weights=True)

# Train the model
batch_size = 128
epochs = 100
history = model.fit(x_train_new, y_train_new, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[es])

# ============================== Plot =========================== #
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['balanced_accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_balanced_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('loss_accuracy_MLP.png')
# ============================== Predictions =========================== #
# Predict the values from the validation dataset
y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred,axis = 1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1)

# ============================== Evaluation =========================== #
# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=classes)
disp.plot()
# save the plot
plt.savefig('confusion_matrix_MLP.png')

# Classification report
print(classification_report(y_true, y_pred_classes))

# Balanced Accuracy
print('Balanced Accuracy:', balanced_accuracy_score(y_true, y_pred_classes))

# ============================== Naive Bayes =========================== #
print('\n============================== Naive Bayes ===========================\n')
# Reshape the data to 2D
x_train_new = x_train_new.reshape(x_train_new.shape[0], 28*28*3)
x_val = x_val.reshape(x_val.shape[0], 28*28*3)
y_train_naive = y_train_new[:,1]
# Create a Gaussian Classifier
gnb = GaussianNB()

# Train the model using the training sets
gnb.fit(x_train_new, y_train_naive)

# Predict the response for test dataset
y_pred = gnb.predict(x_val)

# ============================== Evaluation =========================== #
# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=classes)
disp.plot()
# save the plot
plt.savefig('confusion_matrix_NaiveBayes.png')

# Classification report
print(classification_report(y_true, y_pred))

# Balanced Accuracy
print('Balanced Accuracy:', balanced_accuracy_score(y_true, y_pred))

# ============================== CNN =========================== #
print('\n============================== CNN ===========================\n')
# Reshape the data to 4D
x_train_new = x_train_new.reshape(x_train_new.shape[0], 28, 28, 3)
x_val = x_val.reshape(x_val.shape[0], 28, 28, 3)

# Create CNN with  ResNet50V2 configuration
model = Sequential()
# Input Layer
model.add(InputLayer(input_shape=(28, 28, 3)))
# First Convolutional Layer
model.add(Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Second Convolutional Layer
model.add(Conv2D(filters=128, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))
# Third Convolutional Layer
model.add(Conv2D(filters=256, kernel_size=(3,3), padding="same", activation="relu"))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.2))


# Flattening
model.add(Flatten())
# First Dense Layer
model.add(Dense(512, activation="relu"))
model.add(Dropout(0.2))
# Second Dense Layer
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
# Output Layer
model.add(Dense(2, activation="softmax"))

model.summary()

# create weighted loss function
# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=[BalancedAccuracy()])
es = EarlyStopping(monitor='val_balanced_accuracy', mode='min', verbose=1, patience=10, restore_best_weights=True)

# Train the model
batch_size = 128
epochs = 100
history = model.fit(x_train_new, y_train_new, batch_size=batch_size, epochs=epochs, validation_data=(x_val, y_val), callbacks=[es])

# save the model to disk
model.save('model_cnn.keras')
# ============================== Plot =========================== #
# Plot the loss and accuracy curves for training and validation
fig, ax = plt.subplots(2,1)
ax[0].plot(history.history['loss'], color='b', label="Training loss")
ax[0].plot(history.history['val_loss'], color='r', label="validation loss",axes =ax[0])
legend = ax[0].legend(loc='best', shadow=True)

ax[1].plot(history.history['balanced_accuracy'], color='b', label="Training accuracy")
ax[1].plot(history.history['val_balanced_accuracy'], color='r',label="Validation accuracy")
legend = ax[1].legend(loc='best', shadow=True)
plt.savefig('loss_accuracy_CNN.png')

# ============================== Predictions =========================== #
# Predict the values from the validation dataset
y_pred = model.predict(x_val)
# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred,axis = 1)
# Convert validation observations to one hot vectors
y_true = np.argmax(y_val,axis = 1)

# ============================== Evaluation =========================== #
# Compute the confusion matrix
confusion_mtx = confusion_matrix(y_true, y_pred_classes)
print(confusion_mtx)
# Plot the confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=confusion_mtx, display_labels=classes)
disp.plot()
# save the plot
plt.savefig('confusion_matrix_CNN.png')

# Classification report
print(classification_report(y_true, y_pred_classes))

# Balanced Accuracy
print('Balanced Accuracy:', balanced_accuracy_score(y_true, y_pred_classes))

# ============================== Submission =========================== #

x_test_CNN = x_test.reshape(x_test.shape[0], 28, 28, 3)
y_pred = model.predict(x_test_CNN)
print("Shape of prediction:", y_pred.shape)

# Convert predictions classes to one hot vectors
y_pred_classes = np.argmax(y_pred,axis = 1)
print("Shape of prediction classes:", y_pred_classes.shape)

# print number of nevus and melanoma
nevus = np.count_nonzero(y_pred_classes == 0)
melanoma = np.count_nonzero(y_pred_classes == 1)
print('Total number of nevus:', nevus)
print('Total number of melanoma', melanoma)
# save to npy file
np.save('output.npy', y_pred_classes)