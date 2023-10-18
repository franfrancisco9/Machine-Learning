import numpy as np
import os
import matplotlib.pyplot as plt
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Add, Dropout
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from keras.losses import Loss
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
import keras.backend as K
import tensorflow as tf
import keras
from itertools import product
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

class BalAccScore(keras.callbacks.Callback):

    def __init__(self, validation_data=None):
        super(BalAccScore, self).__init__()
        self.validation_data = validation_data
        
    def on_train_begin(self, logs={}):
      self.balanced_accuracy = []

    def on_epoch_end(self, epoch, logs={}):
        y_predict = tf.argmax(self.model.predict(self.validation_data[0]), axis=1)
        y_true = tf.argmax(self.validation_data[1], axis=1)
        balacc = balanced_accuracy_score(y_true, y_predict)
        self.balanced_accuracy.append(round(balacc,6))
        logs["val_bal_acc"] = balacc
        keys = list(logs.keys())

        print("\n ------ validation balanced accuracy score: %f ------\n" %balacc)
def plot_metrics(history, batch_size, patience, cm):
    # Plot accuracy
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['balanced_accuracy'])
    plt.plot(history.history['val_balanced_accuracy'])
    plt.title('Model accuracy for batch size {} and patience {}'.format(batch_size, patience))
    plt.ylabel('balanced_accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model Loss for batch size {} and patience {}'.format(batch_size, patience))
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')

    plt.tight_layout()
    plt.savefig(f"cnn_images/accuracy_loss_b_{batch_size}_p_{patience}.png")
    plt.close()

    # Plot confusion matrix
    plt.figure(figsize=(10, 10))
    plt.title('Confusion matrix for batch size {} and patience {}'.format(batch_size, patience))
    plt.imshow(cm, cmap=plt.cm.Blues)
    # add the values inside each square
    thresh = cm.max() / 2.
    for i, j in product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.xticks([], [])
    plt.yticks([], [])
    plt.colorbar()
    plt.savefig(f"cnn_images/confusion_matrix_b_{batch_size}_p_{patience}.png")
    plt.close()

class WeightedCategoricalCrossentropy(Loss):
    def __init__(self, weights, **kwargs):
        super().__init__(**kwargs)
        self.weights = weights

    def call(self, y_true, y_pred):
        nb_cl = len(self.weights)
        final_mask = K.zeros_like(y_pred[:, 0])
        y_pred_max = K.max(y_pred, axis=1)
        y_pred_max = K.expand_dims(y_pred_max, 1)
        y_pred_max_mat = K.equal(y_pred, y_pred_max)
        for c_p, c_t in product(range(nb_cl), range(nb_cl)):
            final_mask += (self.weights[c_t, c_p] * K.cast(y_pred_max_mat[:, c_p], 'float32') * K.cast(y_true[:, c_t], 'float32'))
        return K.categorical_crossentropy(y_true, y_pred) * final_mask

    def get_config(self):
        base_config = super().get_config()
        return {**base_config, "weights": self.weights}

def identity_block(X, filters):
    X_shortcut = X
    f1, f2 = filters

    # First component
    X = Conv2D(filters=f1, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization()(X)
    X = Activation('relu')(X)

    # Second component
    X = Conv2D(filters=f2, kernel_size=(3, 3), strides=(1, 1), padding='same')(X)
    X = BatchNormalization()(X)
    
    # Adjust the depth of the shortcut tensor to match the processed tensor
    X_shortcut = Conv2D(f2, (1, 1), strides=(1,1), padding='valid')(X_shortcut)

    # Add the shortcut tensor to the processed tensor
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X

def load_and_preprocess_data():
    x_train_full = np.load("Xtrain_Classification1.npy")
    y_train_full = np.load("ytrain_Classification1.npy")
    x_test_final = np.load("Xtest_Classification1.npy")

    # split data making sure there are 1 labels in the y_test otherwise redo the split
    while True:
        x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
        if 1 in y_test:
            break
    # print the number of zeros and 1s in train and in test
    print("Train: ", np.unique(y_train, return_counts=True))
    print("Test: ", np.unique(y_test, return_counts=True))

    
    x_train = x_train.reshape(-1, 28, 28, 3) / 255.0
    x_test = x_test.reshape(-1, 28, 28, 3) / 255.0
    x_test_final = x_test_final.reshape(-1, 28, 28, 3)

    return x_train, x_test, y_train, y_test, x_test_final

def augment_data(x_train, y_train):
    x_melanoma = x_train[y_train == 1]
    y_melanoma = y_train[y_train == 1]

    datagen = ImageDataGenerator(rotation_range=20, width_shift_range=0.2, height_shift_range=0.2,
                                 shear_range=0.2, zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
    datagen.fit(x_melanoma)

    augmented_data = []
    target_samples = 5 * len(x_melanoma)
    for x_batch, y_batch in datagen.flow(x_melanoma, y_melanoma, batch_size=32):
        augmented_data.append((x_batch, y_batch))
        if len(augmented_data) * 32 > target_samples:
            break

    x_augmented = np.vstack([data[0] for data in augmented_data])
    y_augmented = np.hstack([data[1] for data in augmented_data])

    x_train = np.vstack([x_train, x_augmented])
    y_train = np.hstack([y_train, y_augmented])

    return x_train, y_train

def build_and_train_model(x_train, y_train, x_test, y_test, batch_size, patience):
    w_array = np.array([[1., 1.], [6., 1.]])
    loss = WeightedCategoricalCrossentropy(weights=w_array)
   
    input_shape = (28, 28, 3)
    input_img = Input(shape=input_shape)

    # Initial layers
    X = Conv2D(32, (3, 3), activation = 'relu')(input_img)
    X = MaxPooling2D((2, 2))(X)
    # X = BatchNormalization()(X)

    # Identity blocks
    # X = identity_block(X, [64, 64])
    # X = identity_block(X, [128, 128])
    # X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.2)(X)

    X = Conv2D(64, (3, 3), activation = 'relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.2)(X)

    X = Conv2D(128, (3, 3), activation = 'relu')(X)
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.2)(X)

    # Flatten and fully connected layers
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    # X = Dropout(0.5)(X)
    # X = Dense(128, activation='relu')(X)
    # X = Dropout(0.5)(X)
    output = Dense(len(np.unique(y_train)), activation='softmax')(X)
    balAccScore = BalAccScore(validation_data=(x_test, to_categorical(y_test)))

    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=[BalancedAccuracy()]) #'binary_crossentropy'

    early_stopping = EarlyStopping(monitor='balanced_accuracy' , patience=patience, restore_best_weights=True)
    checkpoint_filepath = 'best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='balanced_accuracy' , save_best_only=True, mode='min')

    history = model.fit(x_train, to_categorical(y_train), epochs=1000, batch_size=batch_size,
              validation_data=(x_test, to_categorical(y_test)), callbacks=[checkpoint, early_stopping, balAccScore])
    model.load_weights(checkpoint_filepath)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    # confusion matrix plot save 
    cm = confusion_matrix(y_test, y_pred_classes)

    return balanced_accuracy_score(y_test, y_pred_classes), model, history, cm

def main():
    x_train, x_test, y_train, y_test, x_test_final = load_and_preprocess_data()
    x_train, y_train = augment_data(x_train, y_train)

    best_bal_acc = 0
    for batch_size in [32, 64, 128, 256]:
        for patience in [5, 10, 15, 20, 25]:
            print("=====================================\n")
            print("Batch size: {}, Patience: {}".format(batch_size, patience))
            print("\n=====================================")
            bal_acc, model, history, cm  = build_and_train_model(x_train, y_train, x_test, y_test, batch_size, patience)
            print("=====================================\n")
            print("Batch size: {}, Patience: {}, Balanced accuracy: {}".format(batch_size, patience, bal_acc))
            print("\n=====================================\n")    
            # Call the plot function
            plot_metrics(history, batch_size, patience, cm)

            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_model = model
                best_batch_size = batch_size
                best_patience = patience
                # save to output.npy x_test_final and y_pred_classes
                print("=====================================\n")
                print("saving to output.npy")
                print("\n=====================================\n")
                y_pred = best_model.predict(x_test_final)
                y_pred_classes = np.argmax(y_pred, axis=1)
                np.save("output.npy", y_pred_classes)
    print("=====================================\n")
    print("=====================================\n")
    print("=====================================\n")
    print("Best batch size: {}, Best patience: {}, Best balanced accuracy: {}".format(best_batch_size, best_patience, best_bal_acc))
    print("\n=====================================\n")
    print("=====================================\n")
    print("=====================================\n")     


if __name__ == "__main__":
    main()