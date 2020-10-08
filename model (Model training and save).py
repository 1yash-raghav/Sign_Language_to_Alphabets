import numpy as np
import pickle
import random
from keras import Model
from keras.layers import Dense, Dropout
from keras.applications import ResNet50

with open('Frames.pickle', 'rb') as handle:  # Reading_pickle_file_to_extract_Frames_and_Labels
    data = pickle.load(handle)
frames = data[0]
labels = data[1]
print(len(frames), len(labels))
print(len(frames), len(labels))

c = list(zip(frames, labels))
random.shuffle(c)
frames, labels = zip(*c)

unique = list(set(labels))

word_2_indices = {val: index for index, val in enumerate(unique)}  # Forming 1hot matrix
indices_2_word = {index: val for index, val in enumerate(unique)}
pred = np.zeros((len(labels), len(word_2_indices)))
for ix in range(len(labels)):
    column = word_2_indices[labels[ix]]
    pred[ix][column] = 1

frames = np.array(frames)

model = ResNet50(include_top=False, weights='imagenet', input_shape=(224, 224, 3), pooling='avg')
model.summary()
fc1 = Dense(768, activation='relu')(model.output)
dp1 = Dropout(0.2)(fc1)
fc2 = Dense(256, activation='relu')(dp1)
dp2 = Dropout(0.2)(fc2)
fout = Dense(29, activation='softmax')(dp2)

new_model = Model(input=model.input, output=fout)
new_model.summary()
for ix in range(171):
    new_model.layers[ix].trainable = False
new_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
hist = new_model.fit(frames, pred, shuffle=True, batch_size=16, epochs=10, validation_split=0.15)

new_model.save('./model_weights.h5')

with open('./dictionary.pickle', 'wb') as handle:
    pickle.dump([word_2_indices, indices_2_word], handle)
