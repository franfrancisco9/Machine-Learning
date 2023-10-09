# Bayes
# K nearest neighbors
# MLP
# CNN
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import numpy as np

# Load the data
X_train = np.load("Xtrain_Classification1.npy")
Y_train = np.load("ytrain_Classification1.npy")
X_test = np.load("Xtest_Classification1.npy")

# print the size of Xtest_Classification_1.npy
print("Size of Test Data: ", X_test.shape)
# print the size of Xtrain_Classification_1.npy
print("Size of Train Data: ", X_train.shape)
# print the size of ytrain_Classification_1.npy
print("Size of Train Label: ", Y_train.shape)
# print the number of positive and negative samples in the training set
print("Number of Positive Samples in Training Set: ", np.sum(Y_train == 1))
print("Number of Negative Samples in Training Set: ", np.sum(Y_train == 0))	

# Bayes Classifier with BAS evaluation at the end
from sklearn.naive_bayes import GaussianNB
print("Bayes Classifier")
clf = GaussianNB()
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_train)

# print the confusion matrix
print("Confusion Matrix: ")
print(confusion_matrix(Y_train, Y_pred))
# print the balanced accuracy score
print("Balanced Accuracy Score: ", balanced_accuracy_score(Y_train, Y_pred))

# KNN Classifier with BAS evaluation at the end
from sklearn.neighbors import KNeighborsClassifier
print("KNN Classifier with 3 neighbors")
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, Y_train)
Y_pred = clf.predict(X_train)

# print the confusion matrix
print("Confusion Matrix: ")
print(confusion_matrix(Y_train, Y_pred))
# print the balanced accuracy score
print("Balanced Accuracy Score: ", balanced_accuracy_score(Y_train, Y_pred))

# # MLP Classifier with BAS evaluation at the end
# from sklearn.neural_network import MLPClassifier
# print("MLP Classifier")
# clf = MLPClassifier()
# clf.fit(X_train, Y_train)
# Y_pred = clf.predict(X_train)

# # print the confusion matrix
# print("Confusion Matrix: ")
# print(confusion_matrix(Y_train, Y_pred))
# # print the balanced accuracy score
# print("Balanced Accuracy Score: ", balanced_accuracy_score(Y_train, Y_pred))
print(X_train[0])
# CNN Classifier with BAS evaluation at the end
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, MaxPooling2D
from keras.utils import to_categorical
print("CNN Classifier")
# create model
model = Sequential()

# X_reconstructed = X_train.reshape(X_train.shape[0], 28, 28, 3)
X_reconstructed = np.load("Xtrain_Classification1_reconstructed.npy")
# generate the images from X_reconstructed into a folder images
# import matplotlib.pyplot as plt
# for i in range(X_reconstructed.shape[0]):
#     plt.imsave("images/image"+str(i)+".png", X_reconstructed[i])
print("Size of Reconstructed Train Data: ", X_reconstructed.shape)
# SAVE THE RECONSTRUCTED DATA
np.save("Xtrain_Classification1_reconstructed", X_reconstructed)
print("CNN Classifier")
# Convert to one-hot encoded
Y_train_onehot = to_categorical(Y_train)
print("Size of One Hot Encoded Train Label: ", Y_train_onehot.shape)
# SAVE THE ONE HOT ENCODED LABEL    
np.save("ytrain_Classification1_onehot", Y_train_onehot)
model.add(Conv2D(64, kernel_size=3, activation='relu', input_shape=(28,28,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=3, activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(2, activation='softmax'))

# Using categorical crossentropy with one-hot encoded labels
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.fit(X_reconstructed, Y_train_onehot, validation_split=0.2, epochs=3)

# Predict on training set
Y_pred = model.predict(X_reconstructed)
# Convert predictions to labels
Y_pred_labels = np.argmax(Y_pred, axis=1)	
# print the confusion matrix
print("Confusion Matrix: ")
print(confusion_matrix(Y_train, Y_pred_labels))
# print the balanced accuracy score
print("Balanced Accuracy Score: ", balanced_accuracy_score(Y_train, Y_pred_labels))