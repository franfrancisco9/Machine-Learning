
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
import keras.backend as K
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
from sklearn.svm import SVC


# Metrics functions
def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

def bal_acc(y_pred, y_true):
    recall = recall_m(y_pred, y_true)
    precision = precision_m(y_pred, y_true)
    return 0.5 * (recall + precision)

x_train = np.load("Xtrain_Classification2.npy")
y_train = np.load("ytrain_Classification2.npy")
x_test = np.load("Xtest_Classification2.npy")

print("Train size", x_train.shape)
print("Labels Size", y_train.shape)
print("Test Size", x_test.shape)

# print the distribution of classes of the training set
unique, counts = np.unique(y_train, return_counts=True)
# 0 is nevu
# 1 is melanoma
# 2 is vascular lesions
# 3 is granulocytes
# 4 is basophils
# 5 is lymphocytes
# 0,1 and 2 come from dermoscopy train set
# 3,4 and 5 come from blood cell mycroscopy train set
# print the percentages of each class and percentages of each original dataset
# each print should be like Percentage of Nevu in the training set: 0.2
print("Percentage of Nevu in the training set:", counts[0] / sum(counts) * 100)
print("Percentage of Melanoma in the training set:", counts[1] / sum(counts) * 100)
print("Percentage of Vascular Lesions in the training set:", counts[2] / sum(counts) * 100)
print("Percentage of Granulocytes in the training set:", counts[3] / sum(counts) * 100)
print("Percentage of Basophils in the training set:", counts[4] / sum(counts) * 100)
print("Percentage of Lymphocytes in the training set:", counts[5] / sum(counts) * 100)
print("Percentage of data from dermoscopy in the training set:", (counts[0] + counts[1] + counts[2]) / sum(counts) * 100)
print("Percentage of data from blood cell mycroscopy in the training set:", (counts[3] + counts[4] + counts[5]) / sum(counts) * 100)
# inside the sum of each origin see the percentage of each class
print("Percentage of Nevu in the dermoscopy training set:", counts[0] / (counts[0] + counts[1] + counts[2]) * 100)
print("Percentage of Melanoma in the dermoscopy training set:", counts[1] / (counts[0] + counts[1] + counts[2]) * 100)
print("Percentage of Vascular Lesions in the dermoscopy training set:", counts[2] / (counts[0] + counts[1] + counts[2]) * 100)
print("Percentage of Granulocytes in the blood cell mycroscopy training set:", counts[3] / (counts[3] + counts[4] + counts[5]) * 100)
print("Percentage of Basophils in the blood cell mycroscopy training set:", counts[4] / (counts[3] + counts[4] + counts[5]) * 100)
print("Percentage of Lymphocytes in the blood cell mycroscopy training set:", counts[5] / (counts[3] + counts[4] + counts[5]) * 100)

# ==== KNN ====
# ==== Bayes ====
# ==== MLP ====
print("==== MLP ====")
# normalize the data
x_train = x_train.reshape(x_train.shape[0], 28, 28, 3)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 3)

# normalize the data
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255.0
x_test /= 255.0

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, 6)

# split the training set into training and validation sets
# the validation set will be used to evaluate the model

x_train_mlp, x_val_mlp, y_train_mlp, y_val_mlp = train_test_split(x_train, y_train, test_size=0.2, random_state=95789)
# define the model
model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 3)))
model.add(Flatten())
model.add(Dense(224, activation='relu'))
model.add(Dense(6, activation='softmax'))

# print the model summary
model.summary()

es = EarlyStopping(monitor='val_bal_acc', mode='max', verbose=1, patience=5)

# compile the model
model.compile(loss='categorical_crossentropy',
                optimizer=RMSprop(),
                metrics=[bal_acc])

# train the model
history = model.fit(x_train_mlp, y_train_mlp,
                    batch_size=128,
                    epochs=50,
                    verbose=1,
                    validation_split=0.2,
                    callbacks=[es])

# evaluate the model on the validation set
score = model.evaluate(x_val_mlp, y_val_mlp, verbose=0)
print('Validation loss:', score[0])
print('Validation balanced accuracy:', score[1])

# plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig('loss_mlp.png')
# plot the training and validation balanced accuracy
plt.plot(history.history['bal_acc'], label='Training balanced accuracy')
plt.plot(history.history['val_bal_acc'], label='Validation balanced accuracy')
plt.legend()
plt.savefig('balanced_accuracy_mlp.png')

# ==== SVM ====
print("==== SVM ====")
x_train_svm = x_train.reshape(x_train.shape[0], 28 * 28 * 3)

x_train_svm, x_val_svm, y_train_svm, y_val_svm = train_test_split(x_train_svm, y_train, test_size=0.2, random_state=95789)
# setup the SVM classifier
clf = SVC(kernel='linear', C=1, random_state=95789)

# train the SVM classifier
clf.fit(x_train_svm, y_train)

# predict the labels of the validation set
y_pred = clf.predict(x_val_svm)

# print the BAC
print("Balanced Accuracy Score: ", balanced_accuracy_score(y_val_svm, y_pred))


# ==== CNN ====
print("==== CNN ====")
# reshape the data to be fed to the neural network
# the first dimension is the number of images
# the second and third dimensions are the dimensions of each image
# the fourth dimension is the number of channels (3 for RGB)
# split the training set into training and validation sets
# the validation set will be used to evaluate the model

x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=95789)

# print the size of the training set and the validation set
print("Train size", x_train.shape)
print("Validation size", x_val.shape)


# define the model
model = Sequential()
model.add(InputLayer(input_shape=(28, 28, 3)))
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu')) # 32 3
model.add(Conv2D(64, kernel_size=(5, 5), activation='relu')) # 64 5
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu')) # 128
model.add(Dropout(0.2))
model.add(Dense(6, activation='softmax'))

# print the model summary
model.summary()
# compile the model
model.compile(loss='categorical_crossentropy',
              optimizer=RMSprop(),
              metrics=[bal_acc])

# early stopping
es = EarlyStopping(monitor='val_bal_acc', mode='max', verbose=1, patience=5)

# train the model
history = model.fit(x_train, y_train,
                    batch_size=128,
                    epochs=50,
                    verbose=1,
                    validation_data=(x_val, y_val),
                    callbacks=[es])

# evaluate the model on the validation set
score = model.evaluate(x_val, y_val, verbose=0)
print('Validation loss:', score[0])
print('Validation balanced accuracy:', score[1])

# plot the training and validation loss
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.legend()
plt.savefig('loss_cnn.png')
# plot the training and validation balanced accuracy
plt.plot(history.history['bal_acc'], label='Training balanced accuracy')
plt.plot(history.history['val_bal_acc'], label='Validation balanced accuracy')
plt.legend()
plt.savefig('balanced_accuracy_cnn.png')

# predict the labels of the test set
y_pred = model.predict(x_test)
# convert the predicted labels to a numpy array
y_pred = np.argmax(y_pred, axis=1)
# print the predicted labels to output.npy
np.save('output.npy', y_pred)
print("Predicted labels saved to output.npy")
print("Shape of output.npy", y_pred.shape)
print("Neuvus predicted:", np.count_nonzero(y_pred == 0), "In percentage:", np.count_nonzero(y_pred == 0) / len(y_pred) * 100, "%")
print("Melanomas predicted:", np.count_nonzero(y_pred == 1), "In percentage:", np.count_nonzero(y_pred == 1) / len(y_pred) * 100, "%")
print("Vascular Lesions predicted:", np.count_nonzero(y_pred == 2), "In percentage:", np.count_nonzero(y_pred == 2) / len(y_pred) * 100, "%")
print("Granulocytes predicted:", np.count_nonzero(y_pred == 3), "In percentage:", np.count_nonzero(y_pred == 3) / len(y_pred) * 100, "%")
print("Basophils predicted:", np.count_nonzero(y_pred == 4), "In percentage:", np.count_nonzero(y_pred == 4) / len(y_pred) * 100, "%")
print("Lymphocytes predicted:", np.count_nonzero(y_pred == 5), "In percentage:", np.count_nonzero(y_pred == 5) / len(y_pred) * 100, "%")


