import tensorflow as tf
import numpy as np
from sklearn.model_selection import train_test_split

train_full_set = np.load('../../data/processed/train_full_set.npy')
train_full_labels = np.load('../../data/processed/train_full_labels.npy')

X_train, X_valid, y_train, y_valid = train_test_split(train_full_set, train_full_labels, 
                                                      test_size=0.05, shuffle=True, random_state=42)

print('X_train.shape =>', X_train.shape)
print('X_valid.shape =>', X_valid.shape)
print('y_train.shape =>', y_train.shape)
print('y_valid.shape =>', y_valid.shape)


model = tf.keras.Sequential([
    
    tf.keras.layers.Flatten(input_shape=[108//2, 40//2]),
    tf.keras.layers.Rescaling(1./255),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(100, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(10, activation='softmax')

])

model.summary()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(
    X_train, y_train,
    validation_data=(X_valid, y_valid),
    epochs=50)

model.save('../../models/MLP')

