import numpy as np
import cv2

cap = cv2.VideoCapture(0)
data =[]
path = './data/'
element = input("Word : ")

count = 0
while(True):
    ret, frame = cap.read()
    if ret == False:
        continue
    count += 1
    if count <= 15 or count % 10 != 0:
        continue
    cv2.imshow('frame', frame)
    data.append(frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    data = np.asarray(data)
    data = data.reshape((data.shape[0], -1))

    np.save(path + element + '.npy', data)
    print("Data Successfully save at " + path + element + '.npy')
cap.release()
cv2.destroyAllWindows()