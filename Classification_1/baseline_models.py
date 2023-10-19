# Bayes
# K nearest neighbors
# MLP
# SVM
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
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


# Import the necessary libraries for figure creation
import matplotlib.pyplot as plt
import seaborn as sns

x_train, x_test, y_train, y_test = train_test_split(X_train, Y_train, test_size=0.2, random_state=42)
# Bayes Classifier Model
from sklearn.naive_bayes import GaussianNB
print("Bayes Classifier")
clf = GaussianNB()
clf.fit(x_train, y_train)
Y_pred = clf.predict(x_test)

conf_matrix = confusion_matrix(y_test, Y_pred)

# Plot and save the confusion matrix figure
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - Bayes Classifier")
plt.savefig("confusion_matrix_bayes.png")  # Save the figure
plt.show()

# Calculate and print the balanced accuracy score
print("Balanced Accuracy Score for Bayes Classifier: ", balanced_accuracy_score(y_test, Y_pred))

# SVM Model

print("SVM Classifier")
clf = SVC()
clf.fit(x_train, y_train)
Y_pred = clf.predict(x_test)

# Calculate the confusion matrix
conf_matrix = confusion_matrix(y_test, Y_pred)

# Print and save the confusion matrix figure
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix - SVM Model')
plt.savefig('svm_confusion_matrix.png')
plt.show()

# Calculate and print the Balanced Accuracy score
bas = balanced_accuracy_score(y_test, Y_pred)
print("Balanced Accuracy Score: ", bas)


# MLP Classifier

from sklearn.neural_network import MLPClassifier
print("MLP Classifier")
clf = MLPClassifier()
clf.fit(x_train, y_train)
Y_pred = clf.predict(x_test)

# Create and save the confusion matrix

conf_matrix = confusion_matrix(y_test, Y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - MLP Classifier")
plt.savefig("confusion_matrix_mlp.png")  # Save the figure
plt.show()
# Calculate and print the Balanced Accuracy score 
print("Balanced Accuracy Score for MLP: ", balanced_accuracy_score(y_test, Y_pred))

# ... Repeat the above code for KNN Classifier ...

# KNN Classifier with 3 neighbors
print("KNN Classifier with 3 neighbors")
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(x_train, y_train)
Y_pred = clf.predict(x_test)

conf_matrix = confusion_matrix(y_test, Y_pred)

# Create and save the confusion matrix 
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - KNN Classifier")
plt.savefig("confusion_matrix_knn.png")  # Save the figure
plt.show()

# Calculate and print the Balanced Accuracy score
print("Balanced Accuracy Score for kNN: ", balanced_accuracy_score(y_test, Y_pred))
