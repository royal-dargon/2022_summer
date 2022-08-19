import os
import imageio
import numpy as np
import cv2
import matplotlib.pyplot as plt


# 这个函数是对data文件中的图像进行提取, 同时也是做了二值化处理
def get_image():
    images = []
    # 获取文件夹下的文件名
    file_path = 'D:\\src\\2022_summer\\mathematical_modeling\\2013B\\data\\'
    files = os.listdir(file_path)
    for file in files:
        file_name = 'D:\\src\\2022_summer\\mathematical_modeling\\2013B\\data\\' + file
        # (180, 72)
        image = imageio.imread(file_name)
        image = cv2.adaptiveThreshold(src=image, maxValue=1, adaptiveMethod=cv2.ADAPTIVE_THRESH_MEAN_C, thresholdType=cv2.THRESH_BINARY, blockSize=11, C=1)
        images.append(image)
    return np.array(images).astype(float)


# 这个函数是为了将图像取出左边界，右边界，上边界，下边界
def data_format(images):
    image_dicts = []
    for image in images:
        left = image[:][0]
        right = image[:][-1]
        up = image[0][:]
        down = image[-1][:]
        image_dict = {
            "left": np.array(left).astype(float),
            "right": np.array(right).astype(float),
            "up": np.array(up).astype(float),
            "down": np.array(down).astype(float)
        }
        image_dicts.append(image_dict)
    return image_dicts


# 这个函数是用来确定图片出现文字的部分到边界的距离
def left_right(images):
    # image.shape(209, 180, 72)
    l_index = []
    for i in range(images.shape[0]):
        count_left = 0
        for n in range(images.shape[2]):
            # 用来表示这一列是否全部是没有内容
            flag = 1
            for k in range(images.shape[1]):
                if images[i][k][n] == 0:
                    flag = 0
                    break
            if flag == 1:
                count_left = count_left + 1
            else:
                break
        left = count_left
        count_right = 0
        for n in range(images.shape[2]-1, -1, -1):
            flag_right = 1
            for k in range(images.shape[1]):
                if images[i][k][n] == 0:
                    flag_right = 0
                    break
            if flag_right == 1:
                count_right += 1
            else:
                break
        right = count_right
    # plt.scatter([i for i in range(len(left))], left)
    # plt.scatter([i for i in range(len(right))], right)
    # plt.show()
        l_index.append([i, left, right])
    return l_index


# 这个函数是用来确定左边界的图片的顺序
def left_boundary(image_dicts):
    n = len(image_dicts)
    left_sum = []
    for i in range(n):
        s = np.sum(image_dicts[i][1])
        left_sum.append([i, s])
    new_sum = sorted(left_sum, key=lambda x: x[1])
    # 这个时候直接取出排在最前面的十一个作为左边界
    left_index = []
    for i in range(11):
        left_index.append(new_sum[n-i-1][0])
    return left_index


# 这是对右边界进行处理的函数
def right_boundary(image_dicts):
    n = len(image_dicts)
    left_sum = []
    for i in range(n):
        s = np.sum(image_dicts[i][2])
        left_sum.append([i, s])
    new_sum = sorted(left_sum, key=lambda x: x[1])
    # 这个时候直接取出排在最前面的十一个作为左边界
    right_index = []
    for i in range(11):
        right_index.append(new_sum[n - i - 1][0])
    return right_index


# 开始进入聚类的部分
def my_cluster(images, left_index):
    images_word = []
    images_white = []
    width = 0
    for i in range(len(images)):
        image_word = []
        image_white = []
        if sum(images[i][0]) == images.shape[2]:
            flag = 0
        else:
            flag = 1
        for n in range(images.shape[1]):
            # 表示这一行出现了字
            if sum(images[i][n]) != images.shape[2]:
                flag_word = 1
            else:
                flag_word = 0
            if flag != flag_word:
                if flag == 0:
                    if width:
                        image_word.append(width)
                else:
                    if width:
                        image_white.append(width)
                width = 0
            else:
                width += 1
        if flag == 0:
            image_word.append(width)
        else:
            image_white.append(width)
        images_white.append(image_white)
        images_word.append(image_word)
    print(images_word)
    print(images_white)


def main():
    images = get_image()
    l_index = left_right(images)
    # image_dicts = data_format(images)
    left_index = left_boundary(l_index)
    right_index = right_boundary(l_index)
    print("最左边一列的图片序号", left_index)
    print("最右边一列的图片序号", right_index)
    my_cluster(images, left_index)


if __name__ == "__main__":
    main()
