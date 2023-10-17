import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Conv2D, MaxPooling2D, Dropout
from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from keras.layers import Input, Add, BatchNormalization, Activation
from keras.models import Model
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
    
    # Adjusting the shortcut path to match dimensions (if needed)
    if X.shape[-1] != X_shortcut.shape[-1]:
        X_shortcut = Conv2D(filters=f2, kernel_size=(1, 1), strides=(1, 1), padding='same')(X_shortcut)
        X_shortcut = BatchNormalization()(X_shortcut)
    
    # Adding shortcut to the main path
    X = Add()([X, X_shortcut])
    X = Activation('relu')(X)
    
    return X
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Define the checkpoint path and file name
checkpoint_filepath = 'best_weights.h5'

# Create the ModelCheckpoint callback
checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    mode='min'
)

# Load the data
x_train = np.load("Xtrain_Classification1.npy")
y_train = np.load("ytrain_Classification1.npy")
x_test_final = np.load("Xtest_Classification1.npy")
# Load the data
x_train_full = np.load("Xtrain_Classification1.npy")
y_train_full = np.load("ytrain_Classification1.npy")

# Split the full training data into train and test(validation) datasets
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)  # 80% train, 20% test(validation)

# Ensure data is reshaped properly for an image input into a CNN
x_train = x_train.reshape(-1, 28, 28, 3)
x_test = x_test.reshape(-1, 28, 28, 3)
x_test_final = x_test_final.reshape(-1, 28, 28, 3)

# Normalize the data to be between 0 and 1
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Separate the melanoma and non-melanoma samples
x_melanoma = x_train[y_train == 1]
y_melanoma = y_train[y_train == 1]

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

# Fit the data generator
datagen.fit(x_melanoma)

# Augment the melanoma class
augmented_data = []
batch_size = 32
for x_batch, y_batch in datagen.flow(x_melanoma, y_melanoma, batch_size=batch_size):
    augmented_data.append((x_batch, y_batch))
    if len(augmented_data) * batch_size > len(x_melanoma) * 5:  # augmenting to have 5 times more samples than original
        break

x_augmented = np.vstack([data[0] for data in augmented_data])
y_augmented = np.hstack([data[1] for data in augmented_data])

x_train = np.vstack([x_train, x_augmented])
y_train = np.hstack([y_train, y_augmented])
# Convert labels to one-hot encoding
num_classes = len(np.unique(y_train))
y_train_one_hot = to_categorical(y_train, num_classes)

# # Create a simple CNN model
# model = Sequential()
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 3)))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dense(num_classes, activation='softmax'))

# # Compile the model
# model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# # Data augmentation
# datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
# datagen.fit(x_train)

# # Train the model
# batch_size = 128
# epochs = 100
# # Train the model

# # Update the model training to use early stopping:
# model.fit(datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
#           epochs=epochs, verbose=1, 
#           validation_data=(x_test, to_categorical(y_test, num_classes)), 
#           callbacks=[checkpoint, early_stopping])

# model.load_weights(checkpoint_filepath)
# # Predict classes using the test set
# y_pred = model.predict(x_test)
# y_pred_classes = np.argmax(y_pred, axis=1)

# # Calculate and print balanced accuracy
# bal_acc = balanced_accuracy_score(y_test, y_pred_classes)
# print(f"Balanced Accuracy: {bal_acc:.4f}")

# Define hyperparameter ranges
batch_sizes = [8, 16, 32, 64, 128, 256]
patiences = [5]

BEST_BAL_ACC = 0
BEST_BATCH_SIZE = None
BEST_PATIENCE = None

for batch_size in batch_sizes:
    for patience in patiences:
        
        print(f"\nTraining with batch_size: {batch_size} and patience: {patience}\n")
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)
        
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
        output = Dense(num_classes, activation='softmax')(X)

        model = Model(inputs=input_img, outputs=output)

        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

        # Data augmentation
        datagen = ImageDataGenerator(rotation_range=10, zoom_range=0.1, width_shift_range=0.1, height_shift_range=0.1)
        datagen.fit(x_train)

        # Train the model
        model.fit(datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
                  epochs=1000,  # Set to a large number (1000) as we rely on early stopping
                  verbose=1, 
                  validation_data=(x_test, to_categorical(y_test, num_classes)), 
                  callbacks=[checkpoint, early_stopping])

        model.load_weights(checkpoint_filepath)       
        # Predict classes using the test set
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)

        # Calculate and print balanced accuracy
        bal_acc = balanced_accuracy_score(y_test, y_pred_classes)
        print("=====================================")
        print(f"Balanced Accuracy: {bal_acc:.4f} for batch size: {batch_size} and patience: {patience}")
        # Calculate and print confusion matrix
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        print("Confusion Matrix:")
        print(conf_matrix)
        print("=====================================")
        # Check if this combination gives a better balanced accuracy
        if bal_acc > BEST_BAL_ACC:
            BEST_BAL_ACC = bal_acc
            BEST_BATCH_SIZE = batch_size
            BEST_PATIENCE = patience
            # save prediction of x_test_final to output.npy
            y_pred_final = model.predict(x_test_final)
            y_pred_final_classes = np.argmax(y_pred_final, axis=1)
            np.save("output.npy", y_pred_final_classes)
            print("=====================================")
            print("output.npy saved!")
            print("=====================================")

print(f"\nBest Balanced Accuracy: {BEST_BAL_ACC:.4f} achieved with batch size: {BEST_BATCH_SIZE} and patience: {BEST_PATIENCE}")

