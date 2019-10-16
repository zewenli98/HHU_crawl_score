# 验证码中的字符, 就不用汉字了
from os.path import join
import random

number = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
ALPHABET = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U',
            'V', 'W', 'X', 'Y', 'Z']

gen_char_set = number + ALPHABET  # 用于生成验证码的数据集
# 有先后的顺序的

# 图像大小
IMAGE_HEIGHT = 20
IMAGE_WIDTH = 60
MAX_CAPTCHA = 4  # 一共是4位

CHAR_SET_LEN = len(gen_char_set)

# rand = []
# random.seed(666)
# for i in range(156):
#     rand.append(i)
# random.shuffle(rand)
# rand.append(156)
# print(rand)

model_path = "./model/"

model_tag = 'crack_hhu_captcha.model'
save_model = join(model_path, model_tag)

# 输出日志 tensorboard监控的内容
tb_log_path = './logs/'
