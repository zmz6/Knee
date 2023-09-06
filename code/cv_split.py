import os
from random import shuffle
import shutil

root = "../data/knee/group"
cv_nums = 6


def get_index(root, state):
    img_indexes = []
    index_file = os.path.join(root, str(state) + '.txt')
    file = open(index_file, 'r')
    for line in file.readlines():
        img_indexes.append(line.strip('\n'))
    return img_indexes


good_indexes = get_index(root, "good")
bad_indexes = get_index(root, "bad")


shuffle(good_indexes)
shuffle(bad_indexes)
good_nums = len(good_indexes)
bad_nums = len(bad_indexes)

for cv in range(1, cv_nums + 1):
    shuffle(good_indexes)
    shuffle(bad_indexes)
    if not os.path.exists(os.path.join(root, "cv", str(cv))):
        os.makedirs(os.path.join(root, "cv", str(cv)))

    with open(os.path.join(root, "cv", str(cv), 'train.txt'), 'w') as f:
        for index in range(1, int(0.6 * good_nums)):
            f.write(str(good_indexes[index]) + '\n')
        for index in range(1, int(0.6 * bad_nums)):
            f.write(str(bad_indexes[index]) + '\n')

    with open(os.path.join(root, "cv", str(cv), 'val.txt'), 'w') as f:
        for index in range(int(0.6 * good_nums), int(0.8 * good_nums)):
            f.write(str(good_indexes[index]) + '\n')
        for index in range(int(0.6 * bad_nums), int(0.8 * bad_nums)):
            f.write(str(bad_indexes[index]) + '\n')

    with open(os.path.join(root, "cv", str(cv), 'test.txt'), 'w') as f:
        for index in range(int(0.8 * good_nums), good_nums):
            f.write(str(good_indexes[index]) + '\n')
        for index in range(int(0.8 * bad_nums), bad_nums):
            f.write(str(bad_indexes[index]) + '\n')

    print("生成 cv:{}".format(str(cv)))

