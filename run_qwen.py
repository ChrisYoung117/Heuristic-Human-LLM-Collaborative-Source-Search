import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import numpy as np
import scipy.io as scio
import algorithm
import envdisplay
from flask_socketio import SocketIO, emit
import math
import numpy
import random
from PIL import Image
import base64
import io
import os
import os
import base64
from openai import OpenAI
import time
from threading import Lock
lock = Lock()
app = Flask(__name__)
socketio = SocketIO(app)

# UNIT = 25
t = 250
Q = 0.5
D = 1
V = 1
a = 1
X_min = 0
Y_min = 0
X_max = 20
Y_max = 20
particle_number = 2000
step_size = 1
size = 1
user = 'user'
userfail = 'userfail'

CANVAS_MAX_SIZE = 500  # 与前端 max-width/max-height 一致
X_max = 20
Y_max = 20
# 动态计算 UNIT（确保地图铺满画布）
UNIT = CANVAS_MAX_SIZE // max(X_max, Y_max)  # 计算结果为 25（500//20）

# 确保保存图像的文件夹存在
IMAGE_SAVE_DIR = 'saved_images'
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)

# qwen
client = OpenAI(
    api_key="Your API Key",
    base_url="Your Base URL",
)

# 获取saved_images文件夹中最新的图片
def get_latest_image_from_folder(folder_path):
    # 获取文件夹中的所有文件
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    if not files:
        raise FileNotFoundError("No images found in the folder.")
    # 按修改时间排序，获取最新的文件
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def find_most_frequent_letter(text):
    # 初始化计数器
    count_A = 0
    count_B = 0
    count_C = 0
    count_D = 0

    # 遍历文本
    for char in text:
        if char == 'A' or char == 'a':
            count_A += 1
        elif char == 'B' or char == 'b':
            count_B += 1
        elif char == 'C' or char == 'c':
            count_C += 1
        elif char == 'D' or char == 'd':
            count_D += 1

    # 找出出现次数最多的字母
    max_count = max(count_A, count_B, count_C, count_D)

    # 判断哪个字母出现次数最多
    if max_count == 0:
        return "None of A, B, C, D appeared in the text."
    elif max_count == count_A:
        return 'A'
    elif max_count == count_B:
        return 'B'
    elif max_count == count_C:
        return 'C'
    elif max_count == count_D:
        return 'D'



# 将图片转换为Base64编码
def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# prompt_vl = """任务说明：
# 如图所示，是一个机器人执行源搜索任务的场景，图中可能存在标记为A的红色方格区域，标记为B的紫色方格区域，标记为C的蓝色方格区域和标记为D的黄色方格区域。
# 请依次对这些大写英文字母所在区域进行描述，每个区域的描述主要内容为，该区域周围绿点的数量，与其他区域相比分为多、中等、少三个等级，以及它周围的黑色区域的数量，与其他区域相比为多、中等、少三个等级。
# 输出示例如下：
# （英文字母） 区域：
# 该区域周围绿点的数量：等级
# 周围的黑色探索区域数量：等级
# """

prompt_vl = """任务说明：
请基于上传的示意图，明确图中存在的备选目标区域字母标记。A区域标志为红色；B区域标志为紫色；C区域标志为蓝色；D区域标志为黄色。
然后再对存在的区域进行依次描述，每个区域的描述主要内容为，该区域距离小绿点密集区域的距离，分为远、中、近三个等级，它周围的黑色未探索区域的密度为高、中、低三个等级。
输出示例如下（假如代表B的紫色区域不存在）{
A 区域：
距离小绿点密集区域的距离：远
周围的黑色探索区域密度：低
C 区域：
距离小绿点密集区域的距离：中
周围的黑色探索区域密度：中
D 区域：
距离小绿点密集区域的距离：近
周围的黑色探索区域密度：高}

请按照输出示例的格式，对图中存在区域的相关信息进行描述
"""


#无需任何其他文字说明。
# 调用多模态大模型API分析图片

def analyze_image_with_api(image_base64):
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_vl},
            {"type": "image_url",
             "image_url": {
                 "url": f"data:image/png;base64,{image_base64}"  # 使用Base64编码的图片
             }}
        ]}]
    )
    print("--------------------------------------------------------------------------------")
    return completion.choices[0].message.content


class Main(object):
    def __init__(self, map_file='./map_20.mat'):
        self.map_file = map_file
        self.T_in_episode = 0
        data = scio.loadmat(self.map_file)
        self.map_all = np.array(data['MAP_all'], dtype=np.int32)  # 修改数据类型为 int32
        self.source_location = data['source_location']
        self.starting_location = data['starting_location']
        scenes_data = scio.loadmat('./' + user + '/scenes_order')
        self.scenes_order = scenes_data['ac_scenes']
        self.result = []
        self.success_num = 0
        self.episode = -1
        # self.episode = 16
        self.run_num = 20
        self.root = envdisplay.Maze({
            'UNIT': UNIT,  # 直接使用全局变量
            'X_max': X_max,
            'Y_max': Y_max,
            'X_min': X_min,
            'Y_min': Y_min,
            'size': size,
            'step_size': step_size
        })
        self.selected_coordinates = set()
        self.reset()

    def reset(self):
        self.episode += 1
        self.T_in_episode = 0
        # self.scenes = self.scenes_order[0, self.episode]
        self.scenes = self.episode
        self.v_x = self.starting_location[self.scenes, 0]
        self.v_y = self.starting_location[self.scenes, 1]
        self.v_x = math.floor(self.v_x) + 0.5
        self.v_y = math.floor(self.v_y) + 0.5
        self.s_x = self.source_location[self.scenes, 0]
        self.s_y = self.source_location[self.scenes, 1]
        self.o_map = self.map_all[:, :, self.scenes]
        self.unknow_map = np.zeros((21, 21))  # 确保未知区域被重置
        self.trajectory = []
        self.de_up_zero = []
        self.particle = []
        self.result_in_episode = []
        self.is_forbidden = []
        self.result_flag = 0
        self.forbidden_flag = 0
        self.manual_flag = 0
        self.pf = algorithm.ParticleFilter()

        self.root = envdisplay.Maze({
            'UNIT': UNIT,
            'X_max': X_max,
            'Y_max': Y_max,
            'X_min': X_min,
            'Y_min': Y_min,
            'size': size,
            'step_size': step_size
        })
        data = self.root.draw_reset(self.v_x, self.v_y, self.s_x, self.s_y, self.o_map, self.pf.x, self.pf.y)
        socketio.emit('draw', data)  # Ensure this line is called
        self.update_unknown_area()
        self.human_control_record = []
        self.human_control_st = 0
        self.already_flag = 0

    def update_unknown_area(self):
        vx = int(math.floor(self.v_x / size))
        vy = int(math.floor(self.v_y / size))
        for i in range(-1, 2):
            for j in range(-1, 2):
                x = self.v_x + i * step_size
                y = self.v_y + j * step_size
                if (x <= X_max) and (x >= X_min) and (y <= Y_max) and (y >= Y_min):
                    id = (vx + i * step_size) * 20 + vy + j * step_size
                    data = [{'type': 'delete', 'id': id}]
                    socketio.emit('draw', data)
                    self.unknow_map[vx + i * step_size, vy + j * step_size] = 1
    #机器人自动寻源主要逻辑
    def step(self):
        self.update_unknown_area()
        self.T_in_episode += 1
        if math.floor(self.v_x / size) == math.floor(self.s_x / size) and math.floor(self.v_y / size) == math.floor(
                self.s_y / size):
            socketio.emit('info', "The source has been found successfully!")
            data = [{
                'type': 'oval',
                'x': UNIT * self.s_x - 6,
                'y': UNIT * self.s_y - 6,
                'width': 12,
                'height': 12,
                'fill': 'red'
            }]
            socketio.emit('draw', data)
            self.success_num += 1
            self.result_flag = 1
            self.result.append([self.T_in_episode, self.starting_location[self.scenes, 0],
                                self.starting_location[self.scenes, 1], self.s_x, self.s_y, self.result_flag])
            self.save_result()
            # socketio.emit('enable_button', ['next'])
            self.selected_coordinates = set()
            self.bf_next()
            return
        if self.T_in_episode > 400:
            socketio.emit('info', "The search mission is fail!")
            self.result_flag = -1
            self.result.append([self.T_in_episode, self.starting_location[self.scenes, 0],
                                self.starting_location[self.scenes, 1], self.s_x, self.s_y, self.result_flag])
            self.selected_coordinates = set()
            self.bf_next()
            self.save_result_fail()
            return
        self.detection = np.random.poisson(algorithm.diff(self.v_x, self.v_y, self.s_x, self.s_y, a, Q, V, D, t))
        data = self.root.draw_detection(self.detection, self.v_x, self.v_y)
        socketio.emit('draw', data)
        self.pf.update(self.v_x, self.v_y, self.detection)
        neff = 1 / sum(np.square(self.pf.weight))
        if neff < 1 * particle_number:
            self.pf.resample()
            self.pf.mcmcStep(self.v_x, self.v_y, self.detection)
        is_forbidden_flag = algorithm.if_forbidden(self.v_x, self.v_y, self.o_map)
        if is_forbidden_flag:
            self.is_forbidden.extend([1])
            self.o_map[int(math.floor(self.v_x / size)), int(math.floor(self.v_y / size))] = -1
            data = self.root.draw_fb(self.v_x, self.v_y)
            socketio.emit('draw', data)
        else:
            self.is_forbidden.extend([0])
        I = algorithm.infotaixs(self.v_x, self.v_y, self.pf)
        II = algorithm.I2II(self.v_x, self.v_y, self.o_map, I)
        want_action = np.argmin(I)
        self.real_action = np.argmin(II)
        data = self.root.draw_action(self.v_x, self.v_y, want_action, self.real_action)
        socketio.emit('draw', data)  # 确保每次步骤都发出绘制事件
        data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
        socketio.emit('draw', data)  # 确保每次步骤都发出绘制事件
        return


    # 机器人自动寻源主要逻辑结合前端按钮启用禁用逻辑
    def bf_algorithm_step(self):
        socketio.emit('disable_buttons', ['continue'])
        socketio.emit('disable_buttons', ['execute'])
        while True:
            self.step()
            data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)  # 确保每次循环开始时都获取最新的绘制数据
            socketio.emit('draw', data)
            if self.result_flag == 0:
                self.manual_flag = algorithm.if_manual(self.T_in_episode, self.trajectory)
                socketio.emit('draw', data)
                if self.manual_flag == 1:

                    self.chosen_trajectory = np.array([int(math.floor(self.v_x / size)),
                                                       int(math.floor(self.v_y / size))]).reshape(1, 2)

                    x_last = main_instance.chosen_trajectory[-1, 0]
                    y_last = main_instance.chosen_trajectory[-1, 1]

                    socketio.emit('info',
                                  "LLM is being connected to the control source")


                    o_map_explore = np.zeros((21, 21), dtype=int)

                    for i in range(21):
                        for j in range(21):
                            if self.unknow_map[i, j] == 0:
                                o_map_explore[i, j] = 255
                            elif self.unknow_map[i, j] == 1:
                                o_map_explore[i, j] = self.o_map[i, j]


                    def classify_coordinates(map_data, robot_pos):
                        x, y = robot_pos
                        left_up = []
                        right_up = []
                        left_down = []
                        right_down = []
                        for p in range(21):
                            for q in range(21):
                                if map_data[p][q] == 1:
                                    if p <= x and q <= y:
                                        left_up.append((p, q))
                                    if p <= x and q >= y:
                                        right_up.append((p, q))
                                    if p >= x and q <= y:
                                        left_down.append((p, q))
                                    if p >= x and q >= y:
                                        right_down.append((p, q))

                        # 定义检查是否与255相邻的函数
                        def is_adjacent_to_255(pos):
                            px, py = pos
                            for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                                nx, ny = px + dx, py + dy
                                if 0 <= nx < 21 and 0 <= ny < 21:
                                    if map_data[nx][ny] == 255:
                                        return True
                            return False

                        # 过滤每个方向的列表，保留与255相邻的坐标
                        left_up = [pos for pos in left_up if is_adjacent_to_255(pos)]
                        right_up = [pos for pos in right_up if is_adjacent_to_255(pos)]
                        left_down = [pos for pos in left_down if is_adjacent_to_255(pos)]
                        right_down = [pos for pos in right_down if is_adjacent_to_255(pos)]

                        # 定义检查是否被0或255包围的函数
                        def is_surrounded_by_obstacles(pos):
                            p, q = pos
                            neighbors = [(p - 1, q), (p + 1, q), (p, q - 1), (p, q + 1)]
                            for neighbor in neighbors:
                                nx, ny = neighbor
                                if 0 <= nx < 21 and 0 <= ny < 21:
                                    if map_data[nx][ny] not in [0, 255]:
                                        return False
                            return True

                        # 过滤掉被0或255包围的坐标
                        left_up = [pos for pos in left_up if not is_surrounded_by_obstacles(pos)]
                        right_up = [pos for pos in right_up if not is_surrounded_by_obstacles(pos)]
                        left_down = [pos for pos in left_down if not is_surrounded_by_obstacles(pos)]
                        right_down = [pos for pos in right_down if not is_surrounded_by_obstacles(pos)]

                        return {
                            'left_up': left_up,
                            'right_up': right_up,
                            'left_down': left_down,
                            'right_down': right_down
                        }

                    coordinates = classify_coordinates(o_map_explore, (x_last, y_last))

                    # 定义四个列表
                    left_top = coordinates['left_up']
                    right_top = coordinates['right_up']
                    left_bottom = coordinates['left_down']
                    right_bottom = coordinates['right_down']

                    # 移除所有已经选择过的坐标
                    left_top = [pos for pos in left_top if pos not in self.selected_coordinates]
                    right_top = [pos for pos in right_top if pos not in self.selected_coordinates]
                    left_bottom = [pos for pos in left_bottom if pos not in self.selected_coordinates]
                    right_bottom = [pos for pos in right_bottom if pos not in self.selected_coordinates]

                    # 定义一个函数来随机取出元组
                    def get_random_tuple(lst):
                        if lst:  # 检查列表是否为空
                            return random.choice(lst)
                        else:
                            return None

                    # 从每个列表中随机取出一个元组
                    A = get_random_tuple(left_top)
                    B = get_random_tuple(right_top)
                    C = get_random_tuple(left_bottom)
                    D = get_random_tuple(right_bottom)


                    template_A = A
                    template_B = B
                    template_C = C
                    template_D = D


                    # Clear the canvas
                    data = [{'type': 'clear'}]
                    socketio.emit('draw', data)

                    # Redraw the map
                    data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
                    socketio.emit('draw', data)

                    options = []
                    if A is not None:
                        p, q = A
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                options.append({
                                    'type': 'rect',
                                    'x': (p + i) * UNIT,
                                    'y': (q + j) * UNIT,
                                    'width': UNIT,
                                    'height': UNIT,
                                    'fill': 'red'
                                })
                        # 只在中心位置绘制字母
                        options.append({
                            'type': 'text',
                            'x': p * UNIT + UNIT / 2,
                            'y': q * UNIT + UNIT / 2 + 10,
                            'text': 'A',
                            'fill': 'white'
                        })
                    if B is not None:
                        p, q = B
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                options.append({
                                    'type': 'rect',
                                    'x': (p + i) * UNIT,
                                    'y': (q + j) * UNIT,
                                    'width': UNIT,
                                    'height': UNIT,
                                    'fill': 'purple'
                                })
                        # 只在中心位置绘制字母
                        options.append({
                            'type': 'text',
                            'x': p * UNIT + UNIT / 2,
                            'y': q * UNIT + UNIT / 2 + 10,
                            'text': 'B',
                            'fill': 'white'
                        })
                    if C is not None:
                        p, q = C
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                options.append({
                                    'type': 'rect',
                                    'x': (p + i) * UNIT,
                                    'y': (q + j) * UNIT,
                                    'width': UNIT,
                                    'height': UNIT,
                                    'fill': 'blue'
                                })
                        # 只在中心位置绘制字母
                        options.append({
                            'type': 'text',
                            'x': p * UNIT + UNIT / 2,
                            'y': q * UNIT + UNIT / 2 + 10,
                            'text': 'C',
                            'fill': 'white'
                        })
                    if D is not None:
                        p, q = D
                        for i in range(-1, 2):
                            for j in range(-1, 2):
                                options.append({
                                    'type': 'rect',
                                    'x': (p + i) * UNIT,
                                    'y': (q + j) * UNIT,
                                    'width': UNIT,
                                    'height': UNIT,
                                    'fill': 'yellow'
                                })
                        # 只在中心位置绘制字母
                        options.append({
                            'type': 'text',
                            'x': p * UNIT + UNIT / 2,
                            'y': q * UNIT + UNIT / 2 + 10,
                            'text': 'D',
                            'fill': 'black'
                        })
                    socketio.emit('draw', options)

                    socketio.emit('capture_canvas')

                    time.sleep(3)

                    image_path = get_latest_image_from_folder('saved_images')
                    print(f"Analyzing image: {image_path}")
                    # 将图片转换为Base64编码
                    image_base64 = image_to_base64(image_path)
                    # 调用API分析图片
                    result_vl = analyze_image_with_api(image_base64)

                    print(result_vl)

                    prompt_llm = f"""请基于多模态模型提供的结构化描述:{result_vl}，按以下规则输出唯一最高优先级区域：
                    核心规则（优先级依次降低）：
                    不存在的区域不考虑；
                    距离小绿点密级区域最近的优先；
                    区域周围黑色未探索区域密度最高的优先；
                    输出要求：
                    仅输出单个区域字母（如“C”），无需任何解释或排序。"""

                    client = OpenAI(
                        api_key="Your API Key",
                        base_url="Your Base URL",
                    )
                    completion = client.chat.completions.create(
                        model="qwen-max",
                        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                                  {'role': 'user', 'content': prompt_llm}]
                    )
                    response_content = completion.choices[0].message.content
                    print("我的选择是：",response_content)

                    result = find_most_frequent_letter(response_content)

                    # print(result)

                    def get_template(result):
                        templates = [
                            ('A', template_A),
                            ('B', template_B),
                            ('C', template_C),
                            ('D', template_D)
                        ]

                        for letter, template in templates:
                            if letter in result:
                                if template is not None:
                                    return template

                                    # 如果所有模板都为None，或者没有匹配的模板，则遍历查找第一个不为None的模板
                        for _, template in templates:
                            if template is not None:
                                return template

                        return None  # 如果所有模板都为None

                    template = get_template(result)

                    if template is not None:
                        self.selected_coordinates.add(template)

                    print("我的选择是：",template,"我当前的位置是：",(x_last, y_last))
                    print(o_map_explore)

                    x_now , y_now = template

                    # 添加检查逻辑：
                    if (
                            main_instance.o_map[x_now, y_now] != 1
                            or main_instance.unknow_map[x_now, y_now] != 1
                    ):
                        print(f"目标点({x_now}, {y_now})不可通行，重新选择路径")
                        continue  # 跳过当前循环，重新选择目标点

                    self.draw_chosen_trajectory = []
                    if main_instance.manual_flag == 1 and main_instance.already_flag == 0:
                        if main_instance.o_map[x_now, y_now] == 1 and main_instance.unknow_map[x_now, y_now] == 1:
                            myAstar = algorithm.AStar((x_last, y_last), (x_now, y_now), main_instance.o_map,
                                                      main_instance.unknow_map)
                            if myAstar.run() == 1:
                                routelist = myAstar.get_minroute()
                                if not routelist:
                                    print("路径规划失败，无法获取路径")
                                    return
                                main_instance.chosen_trajectory = np.array(routelist,
                                                                           dtype=object)  # 转换为object数组以避免int64问题
                                for i in range(1, len(routelist)):
                                    dct = {
                                        'type': 'rectangle',
                                        'x': int(routelist[i][0] * UNIT),  # 确保转换为Python int
                                        'y': int(routelist[i][1] * UNIT),  # 确保转换为Python int
                                        'width': UNIT,
                                        'height': UNIT,
                                        'outline': "cadetblue",
                                        'width': 2,
                                        'tag': 'ct'
                                    }
                                    main_instance.draw_chosen_trajectory.append(dct)
                                    socketio.emit('draw', [dct])  # 发送绘制指令
                                socketio.emit('info', "Click the EXECUTE to execute the search process")
                                main_instance.already_flag = 1
                                self.bf_execute()
                                self.bf_continue()
                            else:
                                print('路径规划失败！')
                    # socketio.emit('disable_buttons', ['execute'])
                    # socketio.emit('info',
                    #               "The searcher has been trapped. Please select and click a Passable area (white area), then click EXECUTE to help it escape")
                    mean_x, mean_y = algorithm.obtain_cluster(self.pf.x, self.pf.y)
                    data = [{
                        'type': 'oval',
                        'x': UNIT * mean_x - 30,
                        'y': UNIT * mean_y - 30,
                        'width': 60,
                        'height': 60,
                        'outline': 'salmon',
                        'dash': (3, 5),
                        'width': 2,
                        'tag': 'mean_xy'
                    }]
                    socketio.emit('draw', data)
                    self.human_control_st = time.time()
                    return
                socketio.emit('draw', data)  # 发送绘制数据到前端
                if self.real_action == 0:
                    self.v_x += step_size
                elif self.real_action == 1:
                    self.v_y += step_size
                elif self.real_action == 2:
                    self.v_x -= step_size
                elif self.real_action == 3:
                    self.v_y -= step_size
                print('episode', self.episode, 'scenes', self.scenes,
                      'success_num', self.success_num, 'T', self.T_in_episode)
                xy = np.vstack((self.pf.x, self.pf.y)).T
                self.particle.append(xy.tolist())
                self.trajectory.append(self.v_x * 20 + self.v_y)
                self.result_in_episode.append([self.episode, self.T_in_episode,
                                               time.time() - self.l_time, self.v_x, self.v_y, self.detection, 0])
                self.l_time = time.time()
            else:
                if self.episode > self.run_num - 2:
                    socketio.emit('info', "All search missions have been completed!", fg='red')
                    socketio.emit('disable_buttons', ['next'])
                return
    #机器人困住以后点击web画布进入此函数（也是Agent接入点）
    @staticmethod
    def cf_board(e):
        if main_instance.manual_flag == 1 and main_instance.already_flag == 0:
            x_now, y_now = main_instance.obtain_area(e)
            x_last = main_instance.chosen_trajectory[-1, 0]
            y_last = main_instance.chosen_trajectory[-1, 1]
            if main_instance.o_map[x_now, y_now] == 1 and main_instance.unknow_map[x_now, y_now] == 1:
                myAstar = algorithm.AStar((x_last, y_last), (x_now, y_now), main_instance.o_map,
                                          main_instance.unknow_map)
                if myAstar.run() == 1:
                    routelist = myAstar.get_minroute()
                    socketio.emit('disable_buttons', ['continue'])
                    socketio.emit('enable_button', ['execute'])
                    main_instance.chosen_trajectory = np.array(routelist, dtype=object)  # 转换为object数组以避免int64问题
                    for i in range(1, len(routelist)):
                        dct = {
                            'type': 'rectangle',
                            'x': int(routelist[i][0] * UNIT),  # 确保转换为Python int
                            'y': int(routelist[i][1] * UNIT),  # 确保转换为Python int
                            'width': UNIT,
                            'height': UNIT,
                            'outline': "cadetblue",
                            'width': 2,
                            'tag': 'ct'
                        }
                        main_instance.draw_chosen_trajectory.append(dct)
                        socketio.emit('draw', [dct])  # 发送绘制指令
                    socketio.emit('info', "Click the EXECUTE to execute the search process")
                    main_instance.already_flag = 1
                else:
                    print('路径规划失败！')
    #获取用户画布点击坐标
    @staticmethod
    def obtain_area(e):
        x = int(math.floor(e['x'] / size / UNIT))
        y = int(math.floor(e['y'] / size / UNIT))
        return x, y

    #恢复机器人自动寻源逻辑
    # @staticmethod
    def bf_continue(self):
        main_instance.human_control_record.append(
            [main_instance.T_in_episode, main_instance.human_control_st, time.time(),
             time.time() - main_instance.human_control_st, 2])  # 1:execute, 2:continue
        socketio.emit('info', "Algorithm is controlling the search process")
        data = [{'type': 'delete', 'tag': 'mean_xy'}]
        socketio.emit('draw', data)
        if main_instance.real_action == 0:
            main_instance.v_x += step_size
        elif main_instance.real_action == 1:
            main_instance.v_y += step_size
        elif main_instance.real_action == 2:
            main_instance.v_x -= step_size
        elif main_instance.real_action == 3:
            main_instance.v_y -= step_size
        data = main_instance.root.draw_update(main_instance.v_x, main_instance.v_y, main_instance.o_map,
                                              main_instance.pf.x, main_instance.pf.y)
        socketio.emit('draw', data)
        print('episode', main_instance.episode, 'scenes', main_instance.scenes,
              'success_num', main_instance.success_num, 'T', main_instance.T_in_episode)
        xy = np.vstack((main_instance.pf.x, main_instance.pf.y)).T
        main_instance.particle.append(xy.tolist())
        main_instance.trajectory.append(main_instance.v_x * 20 + main_instance.v_y)
        main_instance.result_in_episode.append([main_instance.episode, main_instance.T_in_episode,
                                                time.time() - main_instance.l_time, main_instance.v_x,
                                                main_instance.v_y, main_instance.detection, 0])
        main_instance.l_time = time.time()
        main_instance.bf_algorithm_step()

    #开始机器人自动寻源
    @staticmethod
    def bf_start():
        socketio.emit('disable_buttons', ['start'])
        socketio.emit('info', "Algorithm is controlling the search process")
        main_instance.b_time = time.time()
        main_instance.l_time = main_instance.b_time
        main_instance.result_in_episode.append([main_instance.episode, main_instance.T_in_episode,
                                                main_instance.b_time, main_instance.v_x, main_instance.v_y,
                                                0, 0])  # detection, Flag of Human(1) or Algorithm(0)
        main_instance.bf_algorithm_step()
        socketio.emit('enable_button', 'start')

    #开始人工控制寻源
    # @staticmethod
    def bf_execute(self):
        if len(main_instance.draw_chosen_trajectory) > 0:
            main_instance.already_flag = 0
            main_instance.human_control_record.append(
                [main_instance.T_in_episode, main_instance.human_control_st, time.time(),
                 time.time() - main_instance.human_control_st, 1])  # 1:execute, 2:continue
            data = [{'type': 'delete', 'tag': 'mean_xy'}]
            socketio.emit('draw', data)
            socketio.emit('disable_buttons', ['execute'])
            socketio.emit('info', "LLM is controlling the search process")
            for i in range(len(main_instance.draw_chosen_trajectory)):
                v_x = int(math.floor(main_instance.v_x / size))
                v_y = int(math.floor(main_instance.v_y / size))
                vx = main_instance.chosen_trajectory[i + 1, 0]
                vy = main_instance.chosen_trajectory[i + 1, 1]
                dct = main_instance.draw_chosen_trajectory[i]
                socketio.emit('delete', dct)
                if vx - v_x == 1:
                    main_instance.v_x += step_size
                elif vy - v_y == 1:
                    main_instance.v_y += step_size
                elif vx - v_x == -1:
                    main_instance.v_x -= step_size
                elif vy - v_y == -1:
                    main_instance.v_y -= step_size
                data = main_instance.root.draw_update(main_instance.v_x, main_instance.v_y, main_instance.o_map,
                                                      main_instance.pf.x, main_instance.pf.y)
                socketio.emit('draw', data)
                # print('episode', main_instance.episode, 'scenes', main_instance.scenes,
                #       'success_num', main_instance.success_num, 'T', main_instance.T_in_episode)
                xy = np.vstack((main_instance.pf.x, main_instance.pf.y)).T
                main_instance.particle.append(xy.tolist())
                main_instance.trajectory.append(main_instance.v_x * 20 + main_instance.v_y)
                main_instance.result_in_episode.append([main_instance.episode, main_instance.T_in_episode,
                                                        time.time() - main_instance.l_time, main_instance.v_x,
                                                        main_instance.v_y, main_instance.detection, 1])
                main_instance.l_time = time.time()
                main_instance.step()
                if main_instance.result_flag != 0:
                    return

            # self.root.initial_positions.append((self.v_x, self.v_y))  # 新增：保存重新规划后的位置

            main_instance.draw_chosen_trajectory = []
            main_instance.chosen_trajectory = np.array([int(math.floor(main_instance.v_x / size)),
                                                        int(math.floor(main_instance.v_y / size))]).reshape(1, 2)
            socketio.emit('enable_button', ['continue', 'execute'])
            mean_x, mean_y = algorithm.obtain_cluster(main_instance.pf.x, main_instance.pf.y)
            data = [{
                'type': 'oval',
                'x': UNIT * mean_x - 30,
                'y': UNIT * mean_y - 30,
                'width': 60,
                'height': 60,
                'outline': 'salmon',
                'dash': (3, 5),
                'width': 2,
                'tag': 'mean_xy'
            }]
            socketio.emit('draw', data)
            socketio.emit('info',
                          "Please select another Passable area or click the CONTINUE to allow algorithm to continue to control the search process")
            main_instance.human_control_st = time.time()
        else:
            print('hhhhh')

    #进入下一张地图寻源
    # @staticmethod
    def bf_next(next):
        socketio.emit('disable_buttons', ['next'])
        if main_instance.episode > main_instance.run_num - 2:
            socketio.emit('info', "All search missions have been completed!", fg='red')
            return
        main_instance.reset()
        socketio.emit('info', "Algorithm is controlling the search process")
        main_instance.b_time = time.time()
        main_instance.l_time = main_instance.b_time
        main_instance.result_in_episode.append([main_instance.episode, main_instance.T_in_episode,
                                                main_instance.b_time, main_instance.v_x, main_instance.v_y,
                                                0, 0])  # detection, Flag of Human(1) or Algorithm(0)
        main_instance.bf_algorithm_step()

    def save_result(self):
        dataNew = './' + user + '/AC-scenes' + str(self.scenes) + '.mat'
        scio.savemat(dataNew, {'source_location': [self.s_x, self.s_y],
                               'result': self.result_in_episode,
                               'scenes': self.scenes,
                               'human_control_record': self.human_control_record})
    def save_result_fail(self):
        dataNew = './' + userfail + '/AC-scenes' + str(self.scenes) + '.mat'
        scio.savemat(dataNew, {'source_location': [self.s_x, self.s_y],
                               'result': self.result_in_episode,
                               'scenes': self.scenes,
                               'human_control_record': self.human_control_record})
@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def test_connect():
    print('Client connected')

@socketio.on('disconnect')
def test_disconnect():
    print('Client disconnected')
#处理上传地图文件
@socketio.on('upload')
def handle_upload(json):
    global main_instance
    main_instance = Main(map_file='uploaded_map.mat')
    emit('info', "Map uploaded successfully!")
    emit('enable_button', 'start')  # 启用START按钮

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    if file:
        filename = 'uploaded_map.mat'
        file.save(filename)
        global main_instance
        main_instance = Main(map_file=filename)
        return jsonify({'message': 'Map uploaded successfully!'}), 200

#以下全是监听前端点击事件
@socketio.on('click')
def handle_click(json):
    with lock:
        main_instance.cf_board(json)

@socketio.on('start')
def handle_start():
    with lock:
        main_instance.bf_start()

@socketio.on('continue')
def handle_continue():
    global main_instance
    with lock:
        main_instance.bf_continue()

@socketio.on('execute')
def handle_execute():
    global main_instance
    with lock:
        main_instance.bf_execute()

@socketio.on('next')
def handle_next():
    global main_instance
    with lock:
        main_instance.bf_next()

@socketio.on('save_canvas_image')
def handle_save_canvas_image(json):
    image_data = json['image']
    # 去掉Base64编码的前缀
    image_data = image_data.split(',')[1]
    # 将Base64编码的图像数据解码为二进制数据
    image_binary = base64.b64decode(image_data)
    # 将二进制数据转换为图像
    image = Image.open(io.BytesIO(image_binary))
    # 保存图像到指定文件夹
    image_path = os.path.join(IMAGE_SAVE_DIR, f'canvas_{time.time()}.png')
    image.save(image_path)
    print(f"Canvas image saved to {image_path}")

if __name__ == "__main__":
    socketio.run(app,debug=True,allow_unsafe_werkzeug=True)
