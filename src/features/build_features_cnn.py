import numpy as np
import pandas as pd
import cv2

train_df = pd.read_csv('../../data/interim/df_train.csv')
test_df = pd.read_csv('../../data/interim/df_test.csv')

train = []

for i in range(1, 10):
    train.append(train_df[train_df['label'] == i].sample(500, replace=True))
    
sampled = pd.concat(train, axis=0, ignore_index=True)


new_image_width = 40
new_image_height = 108
size = len(sampled)
train_full_labels = sampled['label'].values
train_full_set = np.empty((size, new_image_height, new_image_width, 1), dtype=np.float32)

for idx, path in enumerate(sampled['image_path']):
    img = cv2.imread(path, -1)
    img = cv2.resize(img, (new_image_width, new_image_height), interpolation=cv2.INTER_NEAREST)
    train_full_set[idx] = img.reshape(new_image_height, new_image_width, 1)
    

print('train_full_set.shape =>', train_full_set.shape)
print('train_full_labels.shape =>', train_full_labels.shape)

np.save('../../data/processed/train_full_set_cnn.npy', train_full_set)
np.save('../../data/processed/train_full_labels_cnn.npy', train_full_labels)

test_full_labels = test_df['label'].values
test_full_set = np.empty((900, new_image_height, new_image_width, 1), dtype=np.float32)
for idx, path in enumerate(test_df['image_path']):
    img = cv2.imread(path, -1)
    img = cv2.resize(img, (new_image_width, new_image_height), interpolation=cv2.INTER_NEAREST)
    
    test_full_set[idx] = img.reshape(new_image_height, new_image_width, 1)
    
print('test_full_set.shape =>', test_full_set.shape)
print('test_full_labels.shape =>', test_full_labels.shape)

np.save('../../data/processed/test_full_set_cnn.npy', test_full_set)
np.save('../../data/processed/test_full_labels_cnn.npy', test_full_labels)