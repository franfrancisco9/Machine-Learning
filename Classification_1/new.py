import numpy as np
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
from itertools import product

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

    x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)
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
    X = Conv2D(64, (3, 3), strides=(1, 1), padding='same', activation='relu')(input_img)
    X = MaxPooling2D((2, 2))(X)
    X = BatchNormalization()(X)

    # Identity blocks
    X = identity_block(X, [64, 64])
    X = identity_block(X, [128, 128])
    X = MaxPooling2D((2, 2))(X)
    X = Dropout(0.5)(X)

    # Flatten and fully connected layers
    X = Flatten()(X)
    X = Dense(256, activation='relu')(X)
    X = Dropout(0.5)(X)
    X = Dense(128, activation='relu')(X)
    X = Dropout(0.5)(X)
    output = Dense(len(np.unique(y_train)), activation='softmax')(X)

    model = Model(inputs=input_img, outputs=output)
    model.compile(optimizer=Adam(), loss=loss, metrics=['accuracy'])

    datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
    datagen.fit(x_train)

    early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
    checkpoint_filepath = 'best_weights.h5'
    checkpoint = ModelCheckpoint(filepath=checkpoint_filepath, monitor='val_loss', save_best_only=True, mode='min')

    model.fit(datagen.flow(x_train, to_categorical(y_train), batch_size=batch_size), epochs=1000,
              validation_data=(x_test, to_categorical(y_test)), callbacks=[checkpoint, early_stopping])
    model.load_weights(checkpoint_filepath)

    y_pred = model.predict(x_test)
    y_pred_classes = np.argmax(y_pred, axis=1)

    return balanced_accuracy_score(y_test, y_pred_classes), model

def main():
    x_train, x_test, y_train, y_test, x_test_final = load_and_preprocess_data()
    x_train, y_train = augment_data(x_train, y_train)

    best_bal_acc = 0
    for batch_size in [32, 64, 128, 256, 512]:
        for patience in [5, 10, 15, 20]:
            print("=====================================\n")
            print("Batch size: {}, Patience: {}".format(batch_size, patience))
            print("\n=====================================")
            bal_acc, model = build_and_train_model(x_train, y_train, x_test, y_test, batch_size, patience)
            print("=====================================\n")
            print("Batch size: {}, Patience: {}, Balanced accuracy: {}".format(batch_size, patience, bal_acc))
            print("\n=====================================\n")
            if bal_acc > best_bal_acc:
                best_bal_acc = bal_acc
                best_model = model
                best_batch_size = batch_size
                best_patience = patience
    print("=====================================\n")
    print("=====================================\n")
    print("=====================================\n")
    print("Best batch size: {}, Best patience: {}, Best balanced accuracy: {}".format(best_batch_size, best_patience, best_bal_acc))
    print("\n=====================================\n")
    print("=====================================\n")
    print("=====================================\n")     


if __name__ == "__main__":
    main()