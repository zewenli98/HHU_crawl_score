import tensorflow as tf
from crawl_HHU.cfg import IMAGE_HEIGHT, IMAGE_WIDTH, CHAR_SET_LEN, MAX_CAPTCHA

X = tf.placeholder(tf.float32, [None, IMAGE_HEIGHT * IMAGE_WIDTH]) # 用placeholder，等被调用时再传入值,None表示参数个数不确定
Y = tf.placeholder(tf.float32, [None, MAX_CAPTCHA * CHAR_SET_LEN])
keep_prob = tf.placeholder(tf.float32)  # dropout 防止overfitting


def crack_captcha_cnn(w_alpha=0.01, b_alpha=0.1):
    """
    定义CNN
    cnn在图像大小是2的倍数时性能最高, 如果你用的图像大小不是2的倍数，可以在图像边缘补无用像素。
    np.pad(image,((2,3),(2,2)), 'constant', constant_values=(255,))  # 在图像上补2行，下补3行，左补2行，右补2行
    """

    x = tf.reshape(X, shape=[-1, IMAGE_HEIGHT, IMAGE_WIDTH, 1])

    '''
    conv2d: 二维的卷积神经网络 

    '''
    w_c1 = tf.Variable(w_alpha * tf.random_normal([3, 3, 1, 32])) # patch 3x3, in size 1, out size 32, 厚度:1->32
    b_c1 = tf.Variable(b_alpha * tf.random_normal([32]))
        # x为输入的所有信息，w为权重，strides为步长 [1, x方向的跨度, y方向的跨度, 1]，
        # padding有两种方式：1、VALID将图片宽高缩小 2、SAME图片宽高不变, b是bias
        # 用relu做非线性化处理
    conv1 = tf.nn.relu6(tf.nn.bias_add(tf.nn.conv2d(x, w_c1, strides=[1, 1, 1, 1], padding='SAME'), b_c1)) # out size: 20 * 60 * 32
        # pool为了防止strides太大而丢失信息，有average_pool 和 max_pool(常用)
    conv1 = tf.nn.max_pool(conv1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')               # out size: 10 * 30 * 32
        # drop掉conv1中部分数据，防止overfitting
    conv1 = tf.nn.dropout(conv1, keep_prob)

    # w_c2 = tf.Variable(w_alpha * tf.random_normal([3, 3, 32, 64])) # patch 3x3, in size 32, out size 64, 厚度:32->64
    # b_c2 = tf.Variable(b_alpha * tf.random_normal([64]))
    # conv2 = tf.nn.relu6(tf.nn.bias_add(tf.nn.conv2d(conv1, w_c2, strides=[1, 1, 1, 1], padding='SAME'), b_c2)) # out size: 10 * 30 * 64
    # conv2 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')                   # out size: 5 * 15 * 64
    # conv2 = tf.nn.dropout(conv2, keep_prob)

    # w_c3 = tf.Variable(w_alpha * tf.random_normal([3, 3, 64, 128]))
    # b_c3 = tf.Variable(b_alpha * tf.random_normal([128]))
    # conv3 = tf.nn.relu6(tf.nn.bias_add(tf.nn.conv2d(conv2, w_c3, strides=[1, 1, 1, 1], padding='SAME'), b_c3)) # out size: 5 * 15 * 128
    # conv3 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')                   # out size:  3 * 8 * 128
    # conv3 = tf.nn.dropout(conv3, keep_prob)

    # w_c4 = tf.Variable(w_alpha * tf.random_normal([3, 3, 128, 256]))
    # b_c4 = tf.Variable(b_alpha * tf.random_normal([256]))
    # conv4 = tf.nn.relu6(tf.nn.bias_add(tf.nn.conv2d(conv3, w_c4, strides=[1, 1, 1, 1], padding='SAME'), b_c4))   # out size: 4 * 8 * 256
    # conv4 = tf.nn.max_pool(conv4, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')                     # out size: 2 * 4 * 256
    # conv4 = tf.nn.dropout(conv4, keep_prob)

    # Fully connected layer
    w_d = tf.Variable(w_alpha * tf.random_normal([10 * 30 * 32, 1024])) # 这里的输入是第三层卷积网络的输出，这里输出为1024
    b_d = tf.Variable(b_alpha * tf.random_normal([1024]))
    dense = tf.reshape(conv1, [-1, w_d.get_shape().as_list()[0]])  # 把矩阵[n_samples, 2, 4, 256] --> 一维 [n_samples, 2 * 4 * 256]
    dense = tf.nn.relu6(tf.add(tf.matmul(dense, w_d), b_d))
    dense = tf.nn.dropout(dense, keep_prob) # 防止overfitting

    w_out = tf.Variable(w_alpha * tf.random_normal([1024, MAX_CAPTCHA * CHAR_SET_LEN]))
    b_out = tf.Variable(b_alpha * tf.random_normal([MAX_CAPTCHA * CHAR_SET_LEN]))
    out = tf.add(tf.matmul(dense, w_out), b_out)
    return out
