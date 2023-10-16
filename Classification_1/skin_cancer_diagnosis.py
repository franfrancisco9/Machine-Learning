import cv2
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.svm import SVC

# Load the data
X_train = np.load("Xtrain_Classification1.npy")
Y_train = np.load("ytrain_Classification1.npy")
X_test = np.load("Xtest_Classification1.npy")
X_reconstructed = np.load("Xtrain_Classification1_reconstructed.npy")

def gaussian_filtering(img, kernel_size):
    Imsmooth = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    return Imsmooth

def k_means_clustering(images_list, K):
    all_pixels = np.vstack(images_list).reshape(-1, 3)
    kmeans = KMeans(n_clusters=K, random_state=0, n_init="auto").fit(all_pixels)
    labels = kmeans.labels_
    reshaped_labels = np.array(np.split(labels, len(images_list)))
    centroids = kmeans.cluster_centers_
    return reshaped_labels, centroids

def get_cluster_proportions(labels, K):
    proportions = []
    for label in labels:
        proportion = np.bincount(label, minlength=K) / float(label.size)
        proportions.append(proportion)
    return np.array(proportions)

def feature_extraction(image):
    lbp_features = extract_lbp(image[:,:,0])  # Using only one channel for simplicity
    # glcm_features = extract_glcm(image[:,:,0])  # Using only one channel for simplicity
    return np.hstack([lbp_features])
from skimage.feature import local_binary_pattern
def extract_lbp(image, P=8, R=1):
    lbp = local_binary_pattern(image, P=P, R=R, method='uniform')
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, P + 3), range=(0, P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)  # normalize
    return hist


images = X_reconstructed
ground_labels = Y_train

# Hyperparameters
kernel_size = 5
K = 2

# Step 1: Gaussian Filtering
smoothed_images = [gaussian_filtering(img, kernel_size) for img in images]

# Step 2: K-Means Clustering
labels, centroids = k_means_clustering(smoothed_images, K)

# Extract features from all images
image_features = np.array([feature_extraction(img) for img in smoothed_images])

# Train the classifier
model = SVC(kernel='linear', class_weight='balanced')
model.fit(image_features, ground_labels)
predicted_labels = model.predict(image_features)
classification_accuracy = balanced_accuracy_score(ground_labels, predicted_labels)

print("Classification Accuracy:", classification_accuracy)
print("Confusion Matrix:\n", confusion_matrix(ground_labels, predicted_labels))
