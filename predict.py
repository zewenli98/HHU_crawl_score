import numpy as np
import tensorflow as tf
from PIL import Image
import os
from crawl_HHU.cfg import MAX_CAPTCHA, CHAR_SET_LEN, model_path
from crawl_HHU.cnn_sys import crack_captcha_cnn, X, keep_prob
from crawl_HHU.utils import vec2text, get_clear_bin_image


def hack_function(sess, predict, captcha_image):
    """
    装载完成识别内容后，
    :param sess:
    :param predict:
    :param captcha_image:
    :return:
    """
    text_list = sess.run(predict, feed_dict={X: [captcha_image], keep_prob: 1})
    text = text_list[0].tolist()
    # print(text_list)
    vector = np.zeros(MAX_CAPTCHA * CHAR_SET_LEN)
    i = 0
    for n in text:
        vector[i * CHAR_SET_LEN + n] = 1
        i += 1
    return vec2text(vector)


def batch_hack_captcha(output, predict, saver, image):
    """
    批量生成验证码，然后再批量进行识别
    :return:
    """

    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(model_path))
        image.show()
        image = get_clear_bin_image(image)
        image.show()
        image = np.array(image)
        test_X = image.flatten()
        predict_text = hack_function(sess, predict, test_X).strip()
    return predict_text


if __name__ == '__main__':
    # 定义预测计算图
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    saver = tf.train.Saver()
    # 测试
    for j in range(10):
        # root = "./captcha/"
        # filePath = os.listdir(root)
        # image = Image.open(root + filePath[j])
        image = Image.open("./captcha_test/" + str(j) + ".jpg")
        predict_text = batch_hack_captcha(output, predict, saver, image)
        print(predict_text)

    print('end...')