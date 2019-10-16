import requests
from bs4 import BeautifulSoup
from PIL import Image
from io import BytesIO
import tensorflow as tf
from crawl_HHU.cfg import MAX_CAPTCHA, CHAR_SET_LEN
from crawl_HHU.cnn_sys import crack_captcha_cnn
from crawl_HHU.predict import batch_hack_captcha


class Login(object):
    def __init__(self):
        # 请求头
        self.headers = {
            'Referer': 'http://202.119.113.135/login.jsp',
            'User-Agent':
            'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Mobile Safari/537.36',
            'Host': '202.119.113.135'
        }
        # 登录请求地址
        self.post_url = 'http://202.119.113.135/loginAction.do'
        # 验证码请求地址
        self.yzm_url = 'http://202.119.113.135/validateCodeAction.do'
        # 维持一个session
        self.session = requests.Session()

    # login函数
    def login(self, account, password):
        # 请求验证码并用CNN识别
        img_response = self.session.get(self.yzm_url, headers=self.headers)
        image = Image.open(BytesIO(img_response.content))
        yzm = batch_hack_captcha(output, predict, saver, image)
        # 登录请求数据
        post_data = {
            'zjh': account,
            'mm': password,
            'v_yzm': yzm,
        }
        # 请求登录
        loginResponse = self.session.post(self.post_url, data=post_data, headers=self.headers)
        cookies = requests.utils.dict_from_cookiejar(self.session.cookies)
        a = cookies['JSESSIONID']
        self.headers2 = {
        'Accept': 'text / html, application / xhtml + xml, application / xml;q = 0.9, image / webp, image / apng, * / *;q = 0.8, application / signed - exchange;v = b3',
        'Accept-Encoding': 'gzip, deflate',
        'Accept - Language': 'zh - CN, zh;q = 0.9',
        'Cache - Control': 'no-cache',
        'Connection': 'keep-alive',
        'Cookie': 'JSESSIONID=' + a,
        'Host': '202.119.113.135',
        'Pragma': 'no - cache',
        'Upgrade - Insecure - Requests': '1',
        'User - Agent': 'Mozilla/5.0 (Linux; Android 6.0; Nexus 5 Build/MRA58N) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/73.0.3683.103 Mobile Safari/537.36'
        }
        url_grade = 'http://202.119.113.135/gradeLnAllAction.do?type=ln&oper=qbinfo&lnxndm=2019-2020%D1%A7%C4%EA1(%C1%BD%D1%A7%C6%DA)'
        html2 = requests.post(url_grade, data=post_data, headers=self.headers2)
        soup = BeautifulSoup(html2.text, 'html.parser')
        return loginResponse.text, soup

    def cal_score(self, soup):
        # 保存每学期的名字
        title_list = []
        b_list = soup.find_all('b')
        for b in b_list:
            title_list.append(b.text)

        table_list = soup.find_all("table", attrs={"class": "titleTop2"})
        total_course = [] # 三维数据
        for table in table_list:
            tr_list = table.find_all("tr", attrs={"class": "odd"})
            semester_course = []
            for tr in tr_list:
                td_list = tr.find_all('td')
                per_course = []
                for cnt, td in enumerate(td_list):
                    per_course.append(td.text.replace('\xa0','').replace('                 ','').replace('                ','').replace('            	 ','').replace('            ','').replace('\r', '').replace('\n', ''))
                per_course.append(float(per_course[4]) * get_per_GPA(per_course[6]))
                semester_course.append(per_course)
            total_course.append(semester_course)

        # 打印成绩
        for i, semester in enumerate(total_course):
            print(title_list[i])
            print(semester)
            cur_GPA, cur_avg_score = calculate_GPA_and_avg_score(semester)
            print("本学期GPA:", cur_GPA)
            print("本学期平均分:", cur_avg_score)
            print()

def get_per_GPA(score):
    if score == "优秀": return 5.0
    if score == "良好": return 4.5
    if score == "中等": return 3.5
    if score == "及格": return 2.5
    if score == "不及格": return 0.0
    if float(score) >= 90: return 5.0
    if float(score) >= 85 and float(score) <= 89: return 4.5
    if float(score) >= 80 and float(score) <= 84: return 4.0
    if float(score) >= 75 and float(score) <= 79: return 3.5
    if float(score) >= 70 and float(score) <= 74: return 3.0
    if float(score) >= 65 and float(score) <= 69: return 2.5
    if float(score) >= 60 and float(score) <= 64: return 2.0
    if float(score) <= 59: return 0.0

def get_per_score(score):
    if score == "优秀": return 95
    if score == "良好": return 85
    if score == "中等": return 75
    if score == "及格": return 65
    if score == "不及格": return 55
    else: return float(score)

def calculate_GPA_and_avg_score(semester_course_list):
    # 计算总GPA、分数
    total_score = 0
    total_credit = 0
    avg_score = 0
    cnt_course = 0
    for per in semester_course_list:
        # 计算GPA
        if per[5] == '必修':
            total_score += per[8]
            total_credit += float(per[4])
        # 计算平均分（包含选修课）
        avg_score += get_per_score(per[6])
        cnt_course += 1
    final_GPA = total_score / total_credit
    avg_score /= cnt_course
    return final_GPA, avg_score


if __name__ == "__main__":
    output = crack_captcha_cnn()
    predict = tf.argmax(tf.reshape(output, [-1, MAX_CAPTCHA, CHAR_SET_LEN]), 2)
    saver = tf.train.Saver()
    myLogin = Login()
    act = input("请输入帐号：")
    psw = input("请输入密码：")
    cnt = 20
    while (cnt):
        txt, soup = myLogin.login(account=act, password=psw)
        if "登录" not in txt:
            break
        cnt -= 1
    myLogin.cal_score(soup)