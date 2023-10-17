import numpy as np
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from resnet import ResnetBuilder

# Load the data
x_train_full = np.load("Xtrain_Classification1.npy")
y_train_full = np.load("ytrain_Classification1.npy")
x_test_final = np.load("Xtest_Classification1.npy")

# Split the data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)  

# Reshape and normalize the data
x_train = x_train.reshape(-1, 28, 28, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 3).astype('float32') / 255.0
x_test_final = x_test_final.reshape(-1, 28, 28, 3)

# Augment the melanoma class
x_melanoma = x_train[y_train == 1]
y_melanoma = y_train[y_train == 1]
datagen = ImageDataGenerator(
    rotation_range=20, width_shift_range=0.2, height_shift_range=0.2, shear_range=0.2, 
    zoom_range=0.2, horizontal_flip=True, fill_mode='nearest'
)
datagen.fit(x_melanoma)
augmented_data = [data for data, _ in datagen.flow(x_melanoma, y_melanoma, batch_size=len(x_melanoma) * 5)]
x_augmented, y_augmented = np.vstack(augmented_data), y_melanoma[:len(augmented_data)]

x_train = np.vstack([x_train, x_augmented])
y_train = np.hstack([y_train, y_augmented])
y_train_one_hot = to_categorical(y_train, len(np.unique(y_train)))

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', restore_best_weights=True)
checkpoint = ModelCheckpoint(
    filepath='best_weights.h5', monitor='val_loss', verbose=1, save_best_only=True, mode='min'
)

# Hyperparameters and training
batch_sizes = [8, 16, 32, 64, 128, 256]
patiences = [5]
BEST_BAL_ACC, BEST_BATCH_SIZE, BEST_PATIENCE = 0, None, None

for batch_size in batch_sizes:
    for patience in patiences:
        print(f"\nTraining with batch_size: {batch_size} and patience: {patience}\n")
        
        early_stopping.patience = patience

        # Create the ResNet model using ResnetBuilder
        model = ResnetBuilder.build_resnet(input_shape=(28, 28, 3), num_outputs=len(np.unique(y_train)))
        
        model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
        datagen.fit(x_train)
        model.fit(
            datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
            epochs=1000, verbose=1, validation_data=(x_test, to_categorical(y_test, len(np.unique(y_train)))),
            callbacks=[checkpoint, early_stopping]
        )

        model.load_weights('best_weights.h5')
        y_pred = model.predict(x_test)
        y_pred_classes = np.argmax(y_pred, axis=1)
        bal_acc = balanced_accuracy_score(y_test, y_pred_classes)
        
        print(f"Balanced Accuracy: {bal_acc:.4f} for batch size: {batch_size} and patience: {patience}")
        conf_matrix = confusion_matrix(y_test, y_pred_classes)
        print("Confusion Matrix:")
        print(conf_matrix)
        
        if bal_acc > BEST_BAL_ACC:
            BEST_BAL_ACC = bal_acc
            BEST_BATCH_SIZE = batch_size
            BEST_PATIENCE = patience
            y_pred_final = model.predict(x_test_final)
            np.save("output.npy", np.argmax(y_pred_final, axis=1))
            print("output.npy saved!")

print(f"\nBest Balanced Accuracy: {BEST_BAL_ACC:.4f} achieved with batch size: {BEST_BATCH_SIZE} and patience: {BEST_PATIENCE}")
