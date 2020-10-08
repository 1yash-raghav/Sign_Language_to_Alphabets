import numpy as np
import cv2
import pickle
from tensorflow.keras.models import load_model

model = load_model('./model_weights.h5')
model.summary()

with open('./dictionary.pickle', 'rb') as handle:
    word_2_indices, indices_2_word = pickle.load(handle)
handle.close()

cap = cv2.VideoCapture(0)

skip = 4
count = 0

while True:

    ret, frame = cap.read()

    if not ret:
        continue

    frame = cv2.flip(frame, 1)
    res_frame = frame[100:324, 100:324]

    count += 1
    if count % 4 != 0:
        continue

    output = model.predict(np.expand_dims(res_frame, axis=0))
    idx = np.argmax(output)
    predicted_name = indices_2_word[idx]
    print(predicted_name)
    cv2.rectangle(frame, (100, 100), (324, 324), (0, 255, 100), 0)
    cv2.putText(frame, predicted_name, (80, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 30, 50), 2, cv2.LINE_AA)

    cv2.imshow('Signs', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
