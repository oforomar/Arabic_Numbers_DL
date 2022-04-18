import numpy as np
import pandas as pd
import cv2

train_df = pd.read_csv('../../data/processed/df_train.csv')
test_df = pd.read_csv('../../data/processed/df_test.csv')


NEW_IMAGE_WIDTH = 40//2
NEW_IMAGE_HEIGHT = 108//2
SIZE = len(train_df)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(3,1))
train_full_labels = train_df['label'].values
train_full_set = np.empty((SIZE, NEW_IMAGE_HEIGHT, NEW_IMAGE_WIDTH, 1), dtype=np.float32)

for idx, path in enumerate(train_df['image_path']):
    img = cv2.imread(path, -1)
    img = cv2.resize(img, (NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    final = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    train_full_set[idx] = final.reshape(NEW_IMAGE_HEIGHT, NEW_IMAGE_WIDTH, 1)

print('train_full_set.shape =>', train_full_set.shape)
print('train_full_labels.shape =>', train_full_labels.shape)

np.save('../../data/processed/train_full_set.npy', train_full_set)
np.save('../../data/processed/train_full_labels.npy', train_full_labels)

test_full_labels = test_df['label'].values
test_full_set = np.empty((900, NEW_IMAGE_HEIGHT, NEW_IMAGE_WIDTH, 1), dtype=np.float32)
for idx, path in enumerate(test_df['image_path']):
    img = cv2.imread(path, -1)
    img = cv2.resize(img, (NEW_IMAGE_WIDTH, NEW_IMAGE_HEIGHT), interpolation=cv2.INTER_NEAREST)
    
    opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    final = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)

    test_full_set[idx] = final.reshape(NEW_IMAGE_HEIGHT, NEW_IMAGE_WIDTH, 1)
    
print('test_full_set.shape =>', test_full_set.shape)
print('test_full_labels.shape =>', test_full_labels.shape)

np.save('../../data/processed/test_full_set.npy', test_full_set)
np.save('../../data/processed/test_full_labels.npy', test_full_labels)