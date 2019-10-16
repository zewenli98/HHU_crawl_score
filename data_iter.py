import numpy as np
from PIL import Image, ImageFilter
import os, random
from crawl_HHU.cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA
from crawl_HHU.utils import text2vec, get_clear_bin_image, convert2gray
import matplotlib.pyplot as pyplot


root = "./captcha/"

def get_next_batch(batch_size=64, cnt = 0):
    """
    # 生成一个训练batch
    :param batch_size cnt
    :return:
    """
    batch_x = np.zeros([batch_size, IMAGE_HEIGHT * IMAGE_WIDTH])
    batch_y = np.zeros([batch_size, MAX_CAPTCHA * CHAR_SET_LEN])

    # f = open(root + "mappings.txt", 'r')
    # lines = f.readlines()
    # f.close()

    filePath = os.listdir(root)
    random.shuffle(filePath)
    # print(filePath)

    i = 0
    for j in range(cnt*batch_size, (cnt+1)*batch_size):
        # text = lines[j].split(",")[-1]
        text = filePath[j].split('.')[0]
        # print(text)
        image = Image.open(root + filePath[j])
        # image.show()
        image = get_clear_bin_image(image)
        # image.show()
        image = np.array(image)
        # pyplot.imshow(image)
        # pyplot.show()

        batch_x[i, :] = image.flatten()  # (image.flatten()-128)/128  mean为0
        batch_y[i, :] = text2vec(text)

        # print(len(batch_x[i]))

        i += 1

    return batch_x, batch_y


get_next_batch(2, 0)
# get_next_batch(3, 2)
