import numpy as np
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from tensorflow.keras.applications.resnet50 import ResNet50
from functools import partial
import keras.backend as K

# Load the data
x_train_full = np.load("Xtrain_Classification1.npy")
y_train_full = np.load("ytrain_Classification1.npy")
x_test_final = np.load("Xtest_Classification1.npy")

# Split the data into training and validation sets
x_train, x_test, y_train, y_test = train_test_split(x_train_full, y_train_full, test_size=0.2, random_state=42)

# Reshape and normalize the data
x_train = x_train.reshape(-1, 28, 28, 3).astype('float32') / 255.0
x_test = x_test.reshape(-1, 28, 28, 3).astype('float32') / 255.0
x_test_final = x_test_final.reshape(-1, 28, 28, 3).astype('float32') / 255.0
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
y_train_one_hot = to_categorical(y_train)

# Custom loss for weighted categorical cross-entropy
def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.expand_dims(y_pred_max, 1)
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * K.cast(y_pred_max_mat[:, c_p], 'float32') * K.cast(y_true[:, c_t], 'float32'))
    return K.categorical_crossentropy(y_true, y_pred) * final_mask

# Setting hyperparameters
batch_size = 32
patience_val = 5
w_array = np.ones((2, 2))
w_array[1, 0], w_array[0, 1] = 6, 1
ncce = partial(w_categorical_crossentropy, weights=w_array)

# Model
input_tensor = keras.layers.Input(shape=(28, 28, 3))
base_model = ResNet50(input_tensor=input_tensor, weights='imagenet', include_top=False)
for layer in base_model.layers[:-96]:
    layer.trainable = False

x = base_model.output
x = keras.layers.GlobalAveragePooling2D()(x)
predictions = keras.layers.Dense(2, activation='softmax')(x)
model = keras.models.Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer=Adam(), loss=ncce, metrics=['accuracy'])

# Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=patience_val, restore_best_weights=True)
checkpoint = ModelCheckpoint('best_weights.h5', monitor='val_loss', save_best_only=True, mode='min')

# Training
model.fit(
    datagen.flow(x_train, y_train_one_hot, batch_size=batch_size),
    epochs=20, validation_data=(x_test, to_categorical(y_test)),
    callbacks=[checkpoint, early_stopping]
)

# Evaluation
model.load_weights('best_weights.h5')
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
bal_acc = balanced_accuracy_score(y_test, y_pred_classes)

print(f"Balanced Accuracy: {bal_acc:.4f}")
conf_matrix = confusion_matrix(y_test, y_pred_classes)
print("Confusion Matrix:")
print(conf_matrix)

y_pred_final = model.predict(x_test_final)
np.save("output.npy", np.argmax(y_pred_final, axis=1))
