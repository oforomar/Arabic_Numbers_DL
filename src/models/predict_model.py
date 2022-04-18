import tensorflow as tf
import numpy as np
import pandas as pd

test_full_set = np.load('../../data/processed/test_full_set.npy')
test_full_labels = np.load('../../data/processed/test_full_labels.npy')

test_df = pd.read_csv('../../data/processed/df_test.csv')

model = tf.keras.models.load_model('../../models/MLP')

model.summary()

y_pred = np.argmax(model.predict(test_full_set), axis=-1)

test_df['pred'] = y_pred

misses = test_df[test_df['label'] != test_df['pred']]

print(f'No. of misclassified examples: {len(misses)}')

