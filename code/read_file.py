import pickle
import os
import cv2
import matplotlib.pyplot as plt


def get_annotation():
    with open(os.path.join('../data/knee', 'annotation'), 'rb') as f:
        anno_dict = pickle.load(f)
    return anno_dict


def get_foot_point(a, b, c):
    da = a[1] - b[1]
    db = b[0] - a[0]
    dc = -da * a[0] - db * a[1]
    return (
        (db * db * c[0] - da * db * c[1] - da * dc) / (da * da + db * db),
        (da * da * c[1] - da * db * c[0] - db * dc) / (da * da + db * db)
    )


ano = get_annotation()

index = "R067754823cm"
landmarks = ano[index][0]

img = plt.imread(os.path.join('../data/knee', str(index) + '.jpg'))


foot_landmark = get_foot_point(landmarks[3], landmarks[4], landmarks[0])

foot1 = get_foot_point(landmarks[3], landmarks[4], landmarks[0])
foot2 = get_foot_point(landmarks[0], foot1, landmarks[1])

plt.scatter(landmarks[0][0], y=landmarks[0][1])
plt.scatter(landmarks[4][0], y=landmarks[4][1])
plt.scatter(landmarks[3][0], y=landmarks[3][1])
plt.scatter(int(foot_landmark[0]), int(foot_landmark[1]))
plt.imshow(img)
plt.show()
