import time
from flask import Flask, render_template, request, redirect, url_for, send_from_directory,jsonify
import numpy as np
import scipy.io as scio
import algorithm
import envdisplay
from flask_socketio import SocketIO, emit
import math
import json
import os
import random
import base64
from openai import OpenAI
from PIL import Image
import io

IMAGE_SAVE_DIR = 'saved_images'
if not os.path.exists(IMAGE_SAVE_DIR):
    os.makedirs(IMAGE_SAVE_DIR)
def get_latest_image_from_folder(folder_path):
    files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.png')]
    if not files:
        raise FileNotFoundError("No images found in the folder.")
    latest_file = max(files, key=os.path.getmtime)
    return latest_file

def image_to_base64(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
    
def extract_reasoning(text):
        try:
            data = json.loads(text)
            if 'reasoning' in data:
                return data['reasoning']
            import re
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            if reasoning_match:
                return reasoning_match.group(1)
                
            return "无法提取推理内容"
        except:
            import re
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)"', text)
            if reasoning_match:
                return reasoning_match.group(1)
            
            return "无法提取推理内容"
    
def analyze_text_with_api(text):
    print("文本原始响应:", text)
    client = OpenAI(
        api_key="Your API Key",
        base_url="Your Base URL",
    )
    reasoning = extract_reasoning(text)
    print("推理内容:", reasoning)
    prompt_llm = f"""请基于多模态模型提供的结构化描述:{reasoning}，按以下规则输出两个独立评分：
黑色区域大致占比
▷ 评分映射：
8-10分：黑色占比 ∈ [0%,20%)
5-8分：黑色占比 ∈ [20%,50%)
2-5分：黑色占比 ∈ [50%,80%)
0-2分：黑色占比 ∈ [80%,100%]

绿色粒子聚集分布情况
▷ 评分映射：
0-2分：无明显粒子聚集
2-5分：存在1-2个小规模粒子聚集区域（<5个粒子），但分布较为分散
5-8分：存在多个大规模粒子聚集区域（>10个粒子），且分布较为集中
8-10分：存在极度密集的粒子聚集区域（粒子几乎连成一块）
你必须严格按照以下JSON格式输出，先推理，再评分，不要包含任何其他文字,不要在JSON中使用引号包裹数字：
  "reasoning": "<简要解释评分依据，包含黑色区域占比、最大聚集区域密度和聚集区域数量>",
  "exploration_score": <根据黑色区域占比评分，范围0-10>,
  "exploitation_score": <根据粒子聚集程度评分，范围0-10>
"""
    try:
        completion = client.chat.completions.create(
        model="qwen-max",
        messages=[{'role': 'system', 'content': 'You are a helpful assistant.'},
                    {'role': 'user', 'content': prompt_llm}]
    )
        
        response_text = completion.choices[0].message.content
        print("API文本原始响应:", response_text)
        
        try:
            import re
            json_match = re.search(r'\{[^{]*"exploration_score".*\}', response_text)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = response_text             
            result = json.loads(json_str)
            required_fields = ['exploration_score', 'exploitation_score', 'reasoning']
            if not all(field in result for field in required_fields):
                raise ValueError("Missing required fields in response")              
            result['exploration_score'] = max(0, min(10, int(result['exploration_score'])))
            result['exploitation_score'] = max(0, min(10, int(result['exploitation_score'])))
            
            return json.dumps(result)
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"JSON处理错误: {e}")
            return json.dumps({
                "exploration_score": 5,
                "exploitation_score": 5,
                "reasoning": "模型输出格式错误，使用默认评分"
            })
            
    except Exception as e:
        print(f"API调用错误: {e}")
        return json.dumps({
            "exploration_score": 5,
            "exploitation_score": 5,
            "reasoning": "API调用失败，使用默认评分"
        })
    

def analyze_image_with_api(image_base64):
    client = OpenAI(
        api_key="Your API Key",
        base_url="Your Base URL",
    )
    prompt_vl = """你是一个图像分析专家，你的任务是基于图像分析并描述

【输入要素分析】
图像由以下要素构成：
基底色块：黑/白两种纯色区域，会被绿色粒子覆盖
绿色粒子：离散分布的元素，会覆盖基底色块

【分析维度】
请分别执行以下两个独立分析：
Ⅰ. 基底色块分析：
▷ 测量方法：
将绿色粒子视为其覆盖区域基底色（黑区粒子视为黑，白区粒子视为白）
黑色基底总面积占比（黑色区域占比 = 黑区面积 / 图片总面积）
请得出黑色区域大致占比
▷ 评分映射：
0-2分：Black_ratio ∈ [0%,10%)
2-5分：Black_ratio ∈ [10%,30%)
5-7分：Black_ratio ∈ [30%,50%)
7-9分：Black_ratio ∈ [50%,70%)
9-10分：Black_ratio ∈ [70%,100%]

▷ 评分映射：
0-2分：无明显粒子聚集
2-5分：存在1-2个小规模粒子聚集区域（<5个粒子），但分布较为分散
5-8分：存在多个大规模粒子聚集区域（>10个粒子），且分布较为集中
8-10分：存在极度密集的粒子聚集区域（粒子几乎连成一块）
你必须严格按照以下JSON格式输出，先推理，再评分，不要包含任何其他文字：
{
  "reasoning": "<简要解释评分依据，包含黑色区域占比、最大聚集区域密度和聚集区域数量>",
  "exploration_score": <根据黑色区域占比评分，范围0-10>,
  "exploitation_score": <根据粒子聚集程度评分，范围0-10>
}  
"""   
    completion = client.chat.completions.create(
        model="qwen-vl-max-latest",
        messages=[{"role": "user", "content": [
            {"type": "text", "text": prompt_vl},
            {"type": "image_url",
                "image_url": {
                    "url": f"data:image/png;base64,{image_base64}"
                }}
        ]}],
        response_format={"type": "json_object"}
    )
    response_text = completion.choices[0].message.content
    print("API原始响应:", response_text)
    return response_text
 

app = Flask(__name__)
socketio = SocketIO(app)

UNIT = 25
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

class Main(object):
    def __init__(self, map_file='./map_20.mat', run_id=None):
        self.map_file = map_file
        self.T_in_episode = 0
        data = scio.loadmat(self.map_file)
        self.map_all = np.array(data['MAP_all'], dtype=np.int32)
        self.source_location = data['source_location']
        self.starting_location = data['starting_location']
        scenes_data = scio.loadmat('./user/scenes_order.mat')
        self.scenes_order = scenes_data['ac_scenes']
        self.result = []
        self.success_num = 0
        self.episode = -1
        self.run_num = 20
        self.root = envdisplay.Maze({
            'UNIT': UNIT,
            'X_max': X_max,
            'Y_max': Y_max,
            'X_min': X_min,
            'Y_min': Y_min,
            'size': size,
            'step_size': step_size
        })
        
        self.process_data = {
            'steps': []
        }
        self.user_clicks = []
        self.trapped_state = []
        self.optional_areas = {}
        
        self.run_id = run_id if run_id else self._get_next_run_id()
        
        self.output_dir = f'run_{self.run_id}/output'
        os.makedirs(self.output_dir, exist_ok=True)     
        self.completed_scenes = set()
        self.load_completed_scenes()       
        self.visit_counts = np.zeros((20, 20))    
        self.reset()

    def _get_next_run_id(self):
        existing_runs = []
        for dirname in os.listdir('.'):
            if dirname.startswith('run_') and os.path.isdir(dirname):
                try:
                    run_num = int(dirname.split('_')[1])
                    output_dir = f'{dirname}/output'
                    if os.path.exists(output_dir):
                        completed_scenes = set()
                        for filename in os.listdir(output_dir):
                            if filename.startswith('complete_process_episode_'):
                                try:
                                    scene_num = int(filename.split('_scene_')[1].split('.')[0])
                                    completed_scenes.add(scene_num)
                                except:
                                    continue
                        if len(completed_scenes) < 20:
                            return run_num
                    existing_runs.append(run_num)
                except:
                    continue
        return max(existing_runs + [0]) + 1

    def load_completed_scenes(self):
        self.completed_scenes = set()
        if os.path.exists(self.output_dir):
            for filename in os.listdir(self.output_dir):
                if filename.startswith('complete_process_episode_'):
                    try:
                        scene_num = int(filename.split('_scene_')[1].split('.')[0])
                        self.completed_scenes.add(scene_num)
                    except:
                        continue
        print(f"已完成的场景: {sorted(list(self.completed_scenes))}")
        print(f"当前运行: run_{self.run_id}")

    def reset(self):
        self.episode += 1     
        if self.episode >= len(self.scenes_order[0]):
            return True
        
        while self.episode < len(self.scenes_order[0]):
            self.scenes = self.scenes_order[0, self.episode]
            if self.scenes not in self.completed_scenes:
                break
            print(f"跳过已完成的场景 {self.scenes}")
            self.episode += 1
        
        self.visit_counts = np.zeros((20, 20))
        
        self.T_in_episode = 0
        self.v_x = self.starting_location[self.scenes, 0]
        self.v_y = self.starting_location[self.scenes, 1]
        self.v_x = math.floor(self.v_x) + 0.5
        self.v_y = math.floor(self.v_y) + 0.5
        self.s_x = self.source_location[self.scenes, 0]
        self.s_y = self.source_location[self.scenes, 1]
        self.o_map = self.map_all[:, :, self.scenes]
        self.unknow_map = np.zeros((20, 20))
        self.trajectory = []
        self.de_up_zero = []
        self.particle = []
        self.result_in_episode = []
        self.is_forbidden = []
        self.result_flag = 0
        self.forbidden_flag = 0
        self.manual_flag = 0
        self.pf = algorithm.ParticleFilter()
        data = self.root.draw_reset(self.v_x, self.v_y, self.s_x, self.s_y, self.o_map, self.pf.x, self.pf.y)
        socketio.emit('draw', data)
        self.update_unknown_area()
        self.human_control_record = []
        self.human_control_st = 0
        self.already_flag = 0
        
        return True

    def record_trapped_state(self, current_position = None, step_id = None, map_info = None, user_click = None, result_in_episode = None):
        trapped_data = {
            'current_position': current_position,
            'step_id': step_id,
            'map_info': map_info,
            'user_click': user_click,
            'result': result_in_episode
        }
        trapped_data = {k: v for k, v in trapped_data.items() if v is not None}
        self.trapped_state.append(trapped_data)
        print("Trapped state recorded.")

    def record_step(self, step_id, current_position, map_info=None, user_click=None):
        step_data = {
            'step_id': step_id,
            'current_position': current_position,
            'map_info': map_info
        }

        if user_click:
            step_data['user_click'] = user_click

        self.process_data['steps'].append(step_data)

        print(f"Step {step_id} recorded.")

    def save_process_data(self):
        os.makedirs(self.output_dir, exist_ok=True)
        filename = f'{self.output_dir}/complete_process_episode_{self.episode}_scene_{self.scenes}.json'
        process_data_with_clicks = {
            'trapped_state': self.trapped_state
        }
        with open(filename, 'w') as f:
            json.dump(process_data_with_clicks, f, indent=4)
        print(f"Complete process saved as {filename}")

    def count_particles_grid(self):
        grid = np.zeros((20, 20), dtype=int)
        for x, y in zip(self.pf.x, self.pf.y):
            i = int(math.floor(x))
            j = int(math.floor(y))
            if 0 <= i < 20 and 0 <= j < 20:
                grid[i, j] += 1
        return grid

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
            main_instance.record_trapped_state(result_in_episode=main_instance.T_in_episode)
            main_instance.save_process_data()
            main_instance.process_data = {'steps': [], 'trapped_state': []}
            main_instance.trapped_state = []
            data = [{
                'type': 'oval',
                'x': UNIT * self.s_x - 6,
                'y': UNIT * self.s_y - 6,
                'width': 12,
                'height': 12,
                'fill': 'red'
            }]
            socketio.emit('draw', data)
            main_instance.bf_next()
            self.success_num += 1
            self.result_flag = 1
            self.result.append([self.T_in_episode, self.starting_location[self.scenes, 0],
                                self.starting_location[self.scenes, 1], self.s_x, self.s_y, self.result_flag])
            self.save_result()
            socketio.emit('enable_button', ['next'])
            self.completed_scenes.add(self.scenes)
            return
        if self.T_in_episode > 400:
            socketio.emit('info', "The search mission is fail!")
            self.result_flag = -1
            self.result.append([self.T_in_episode, self.starting_location[self.scenes, 0],
                                self.starting_location[self.scenes, 1], self.s_x, self.s_y, self.result_flag])
            self.save_result()
            main_instance.bf_next()
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
        socketio.emit('draw', data)
        data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
        socketio.emit('draw', data)
        return

    # 新增方法：随机选择一个可通行区域
    def select_random_passable_area(self):
        current_x = int(math.floor(self.v_x / size))
        current_y = int(math.floor(self.v_y / size))
        
        passable_areas = []
        
        if not hasattr(self, 'selected_points'):
            self.selected_points = set()
        
        map_height, map_width = self.o_map.shape
        for x in range(min(20, map_height)):
            for y in range(min(20, map_width)):
                # 检查条件：
                # 1. 是已知区域(unknow_map=1)
                # 2. 是可通行区域(o_map=1)
                # 3. 不是当前位置附近
                # 4. 不是之前选择过的点
                # 5. 不是被标记为禁止的区域(o_map!=-1)
                if (self.unknow_map[x, y] == 1 and 
                    self.o_map[x, y] == 1 and 
                    not (abs(x - current_x) <= 1 and abs(y - current_y) <= 1) and
                    (x, y) not in self.selected_points and
                    self.o_map[x, y] != -1):
                    
                    myAstar = algorithm.AStar((current_x, current_y), (x, y), self.o_map, self.unknow_map)
                    if myAstar.run() == 1:
                        passable_areas.append((x, y))
        
        if passable_areas:
            chosen_point = random.choice(passable_areas)
            self.selected_points.add(chosen_point)
            return chosen_point
        
        return None

    def select_area_by_particles(self, skip_visual_assessment=False):
        """根据粒子分布选择目标区域，动态平衡探索与利用"""
        current_x = int(math.floor(self.v_x / size))
        current_y = int(math.floor(self.v_y / size))
        
        self.visit_counts[current_x, current_y] += 1
        
        particle_weight = 0.7
        exploration_weight = 0.3
        if not skip_visual_assessment:
            try:
                data = [{'type': 'clear'}]
                socketio.emit('draw', data)
                
                data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
                socketio.emit('draw', data)
                
                socketio.emit('capture_canvas')
                
                time.sleep(3)
                
                image_path = get_latest_image_from_folder('saved_images')
                print(f"分析图片: {image_path}")
                
                image_base64 = image_to_base64(image_path)
                
                result_vl = analyze_image_with_api(image_base64)
                result_text = analyze_text_with_api(result_vl)
                text_result = json.loads(result_text)
                
                exploration_score = text_result.get('exploration_score', 5)
                exploitation_score = text_result.get('exploitation_score', 5)

                total_score = exploration_score + exploitation_score
                if total_score == 0:
                    particle_weight = 0.5
                    exploration_weight = 0.5
                else:
                    particle_weight = min(0.9, max(0.1, exploitation_score / total_score))
                    exploration_weight = 1.0 - particle_weight
                
                print(f"动态权重: 粒子权重={particle_weight:.2f}, 探索权重={exploration_weight:.2f}")
                
            except Exception as e:
                print(f"Visual assignment error, use default weights: {e}")
                total_cells = 20 * 20
                explored_cells = np.sum(self.unknow_map)
                exploration_ratio = explored_cells / total_cells               
                particle_weight = max(0.3, 0.9 - exploration_ratio * 0.6)
                exploration_weight = 1.0 - particle_weight
                
        particle_grid = self.count_particles_grid()
        
        particle_centers = []
        for x in range(2, 18):
            for y in range(2, 18):
                window_particles = sum(
                    particle_grid[i][j] 
                    for i in range(max(0, x-2), min(20, x+3)) 
                    for j in range(max(0, y-2), min(20, y+3))
                )
                particle_centers.append(((x, y), window_particles))
        
        particle_centers.sort(key=lambda x: x[1], reverse=True)
        top_centers = particle_centers[:3] if len(particle_centers) >= 3 else particle_centers
        
        if not top_centers:
            return None
            
        selected_center = random.choice(top_centers)
        particle_center = selected_center[0]        
        candidate_areas = []
        for x in range(20):
            for y in range(20):
                # 1. 检查是否是已知区域
                # 2. 检查是否是可通行区域(o_map=1)
                # 3. 不是被标记为禁止的区域(o_map!=-1)
                if (self.unknow_map[x, y] == 1 and 
                    self.o_map[x, y] == 1 and
                    self.o_map[x, y] != -1):
                    
                    distance_to_current = np.sqrt((x - current_x)**2 + (y - current_y)**2)
                    if distance_to_current < 1:
                        continue
                    
                    neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
                    is_edge = False
                    
                    for nx, ny in neighbors:
                        if (0 <= nx < 20 and 
                            0 <= ny < 20 and 
                            self.unknow_map[nx, ny] == 0):
                            is_edge = True
                            break
                    
                    if is_edge:
                        myAstar = algorithm.AStar((current_x, current_y), (x, y), 
                                            self.o_map, self.unknow_map)
                        if myAstar.run() == 1:
                            exploration_count = 0
                            for i in range(max(0, x-2), min(20, x+2)):
                                for j in range(max(0, y-2), min(20, y+2)):
                                    exploration_count += self.visit_counts[i, j]                           
                            dx = x - particle_center[0]
                            dy = y - particle_center[1]
                            distance_to_particles = np.sqrt(dx*dx + dy*dy)                          
                            particle_score = -particle_weight * distance_to_particles
                            exploration_score = -exploration_weight * exploration_count
                            total_score = particle_score + exploration_score                     
                            candidate_areas.append(((x, y), total_score))
        
        if not candidate_areas:
            return None
        
        candidate_areas.sort(key=lambda x: x[1], reverse=True)        
        top_n = min(3, len(candidate_areas))
        selected_index = random.randint(0, top_n-1)       
        selected_area = candidate_areas[selected_index][0]
        print(f"选择区域: {selected_area}, 得分: {candidate_areas[selected_index][1]:.2f}")
        
        return selected_area
    
    def auto_escape_trap(self, skip_visual_assessment=False):
        best_area = self.select_area_by_particles(skip_visual_assessment=True)
        if best_area is None:
            print('没有找到可通行区域！')
            return False
            
        x_now, y_now = best_area
        x_last = int(math.floor(self.v_x / size))
        y_last = int(math.floor(self.v_y / size))
        # 使用A*算法规划路径
        myAstar = algorithm.AStar((x_last, y_last), (x_now, y_now), self.o_map, self.unknow_map)
        if myAstar.run() == 1:
            routelist = myAstar.get_minroute()
            self.chosen_trajectory = np.array(routelist, dtype=object)   
            self.draw_chosen_trajectory = []
            for i in range(1, len(routelist)):
                dct = {
                    'type': 'rectangle',
                    'x': int(routelist[i][0] * UNIT),
                    'y': int(routelist[i][1] * UNIT),
                    'width': UNIT,
                    'height': UNIT,
                    'outline': "cadetblue",
                    'width': 2,
                    'tag': 'ct'
                }
                self.draw_chosen_trajectory.append(dct)
                socketio.emit('draw', [dct])
            
            user_click = {'x': x_now, 'y': y_now}
            self.user_clicks.append(user_click)
            current_position = {'x': self.v_x, 'y': self.v_y}
            step_id = self.process_data['steps'][-1]['step_id']
            map_info = self.process_data['steps'][-1]['map_info']
            self.record_trapped_state(current_position=current_position, step_id=step_id, 
                                    map_info=map_info, user_click=user_click, result_in_episode=None)
            self.save_process_data()            
            socketio.emit('info', "Auto-selecting passable area and executing search process")
            main_instance.bf_execute()
            return True
        else:
            print('路径规划失败！')
            return False

    # 机器人自动寻源主要逻辑结合前端按钮启用禁用逻辑
    def bf_algorithm_step(self):
        socketio.emit('disable_buttons', ['continue'])
        socketio.emit('disable_buttons', ['execute'])
        while True:
            self.step()
            data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
            socketio.emit('draw', data)
            if self.result_flag == 0:
                self.manual_flag = algorithm.if_manual(self.T_in_episode, self.trajectory)
                socketio.emit('draw', data)
                if self.manual_flag == 1:
                    self.chosen_trajectory = np.array([int(math.floor(self.v_x / size)),
                                                       int(math.floor(self.v_y / size))]).reshape(1, 2)
                    self.draw_chosen_trajectory = []
                    current_position = {'x': self.v_x, 'y': self.v_y}
                    map_info = {
                        'robot_x': self.v_x,
                        'robot_y': self.v_y,
                        'obstacles': self.o_map.tolist(),
                        'unknown_area': self.unknow_map.tolist(),
                        'known_area': np.ones_like(self.o_map).tolist(),
                        'timestamp': time.strftime("%Y-%m-%d %H:%M:%S"),
                        'particle_grid': main_instance.count_particles_grid().tolist(),
                        'gas_concentration': self.detection if isinstance(self.detection, int) else self.detection.tolist()  # Handle detection type
                    }
                    
                    step_id = f'{self.scenes:02}{self.T_in_episode:03}'
                    self.record_step(step_id, current_position, map_info)
                    socketio.emit('info', "Using MLLM to analyze the current state...")                    
                    data = [{'type': 'clear'}]
                    socketio.emit('draw', data)                    
                    data = self.root.draw_update(self.v_x, self.v_y, self.o_map, self.pf.x, self.pf.y)
                    socketio.emit('draw', data)                    
                    socketio.emit('capture_canvas')                   
                    time.sleep(3)
                    
                    try:
                        image_path = get_latest_image_from_folder('saved_images')
                        print(f"分析图片: {image_path}")     
                        image_base64 = image_to_base64(image_path)
                        result_vl = analyze_image_with_api(image_base64)
                        print(f"视觉大模型评估结果: {result_vl}")
                        result_text = analyze_text_with_api(result_vl)
                        print(f"文本分析结果: {result_text}")
                        text_result = json.loads(result_text)
                        exploration_score = text_result.get('exploration_score', 5)
                        exploitation_score = text_result.get('exploitation_score', 5)
                        auto_escape = False
                        if exploration_score > exploitation_score * 1.5:
                            socketio.emit('info', f"视觉模型建议更多探索 (探索分数: {exploration_score}, 开发分数: {exploitation_score})")
                            auto_escape = True
                        elif exploitation_score > exploration_score * 1.5:
                            socketio.emit('info', f"视觉模型建议集中开发 (探索分数: {exploration_score}, 开发分数: {exploitation_score})")
                            auto_escape = True
                        
                        if auto_escape:
                            socketio.emit('info', "Executing the automatic intervention recommended by MLLM...")
                            self.auto_escape_trap(skip_visual_assessment=True)
                            return
                    except Exception as e:
                        print(f"Visual assessment error: {e}")
                                        
                    socketio.emit('disable_buttons', ['execute'])
                    socketio.emit('info',
                                  "The searcher has been trapped. Please select and click a Passable area (white area), then click EXECUTE to help it escape")
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
                    self.auto_escape_trap(skip_visual_assessment=True)
                    if not self.auto_escape_trap(skip_visual_assessment=True):
                        socketio.emit('info',
                                    "Automatic escape failed. Please select and click a Passable area (white area), then click EXECUTE to help it escape")
                    
                    return
                    
                socketio.emit('draw', data)
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
            if e is not None:
                x_now, y_now = main_instance.obtain_area(e)
            else:
                best_area = main_instance.select_area_by_particles()
                if best_area is None:
                    print('没有找到可通行区域！')
                    return
                x_now, y_now = best_area
                
            x_last = main_instance.chosen_trajectory[-1, 0]
            y_last = main_instance.chosen_trajectory[-1, 1]
            if main_instance.o_map[x_now, y_now] == 1 and main_instance.unknow_map[x_now, y_now] == 1:
                myAstar = algorithm.AStar((x_last, y_last), (x_now, y_now), main_instance.o_map,
                                          main_instance.unknow_map)
                if myAstar.run() == 1:
                    routelist = myAstar.get_minroute()
                    socketio.emit('disable_buttons', ['continue'])
                    socketio.emit('enable_button', ['execute'])
                    main_instance.chosen_trajectory = np.array(routelist, dtype=object)
                    for i in range(1, len(routelist)):
                        dct = {
                            'type': 'rectangle',
                            'x': int(routelist[i][0] * UNIT),
                            'y': int(routelist[i][1] * UNIT),
                            'width': UNIT,
                            'height': UNIT,
                            'outline': "cadetblue",
                            'width': 2,
                            'tag': 'ct'
                        }
                        main_instance.draw_chosen_trajectory.append(dct)
                        socketio.emit('draw', [dct])                    
                    if e is None:
                        socketio.emit('info', "Auto-selecting passable area and executing search process")
                        main_instance.bf_execute()
                    else:
                        socketio.emit('info', "Click the EXECUTE to execute the search process")
                        
                    main_instance.already_flag = 1
                    user_click = {'x': x_now, 'y': y_now}
                    main_instance.user_clicks.append(user_click)
                    step_id = f'{main_instance.episode:02}{main_instance.T_in_episode:02}'
                    current_position = {'x': main_instance.v_x, 'y': main_instance.v_y}
                    current_position = {'x': main_instance.v_x, 'y': main_instance.v_y}
                    map_info = {
                    'obstacles': main_instance.o_map.tolist(),
                    'unknown_area': main_instance.unknow_map.tolist(),
                    'known_area': np.ones_like(main_instance.o_map).tolist(),
                    'gas_concentration': main_instance.detection if isinstance(main_instance.detection, int) else main_instance.detection.tolist()
                    } 
                    main_instance.record_trapped_state(current_position = current_position, step_id = main_instance.process_data['steps'][-1]['step_id'], map_info=main_instance.process_data['steps'][-1]['map_info'], user_click= user_click, result_in_episode = None)
                    main_instance.save_process_data()
                    with open(f'user_clicks_episode_{main_instance.episode}_scene_{main_instance.scenes}.json', 'w') as f:
                        json.dump(user_click, f)
                else:
                    print('路径规划失败！')
    @staticmethod
    def obtain_area(e):
        x = int(math.floor(e['x'] / size / UNIT))
        y = int(math.floor(e['y'] / size / UNIT))
        return x, y

    @staticmethod
    def bf_continue():
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
                                                0, 0])
        main_instance.bf_algorithm_step()
        socketio.emit('enable_button', 'start')

    #开始人工控制寻源
    @staticmethod
    def bf_execute():
        if len(main_instance.draw_chosen_trajectory) > 0:
            main_instance.already_flag = 0
            main_instance.human_control_record.append(
                [main_instance.T_in_episode, main_instance.human_control_st, time.time(),
                 time.time() - main_instance.human_control_st, 1])
            data = [{'type': 'delete', 'tag': 'mean_xy'}]
            socketio.emit('draw', data)
            socketio.emit('disable_buttons', ['execute'])
            socketio.emit('info', "Auto-executing search process")
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
            socketio.emit('info', "Algorithm is controlling the search process")
            main_instance.bf_continue()
        else:
            print('hhhhh')

    def bf_next(self):
        """进入下一张地图寻源"""
        socketio.emit('disable_buttons', ['next'])
        if self.episode > self.run_num - 2:
            print("当前run完成，开始下一个run")
            self.save_result()
            self.episode = -1
            self.success_num = 0
            self.completed_scenes = set()
            
            self.run_id = self._get_next_run_id()
            self.output_dir = f'run_{self.run_id}/output'
            os.makedirs(self.output_dir, exist_ok=True)
            
            print(f"开始新的运行: run_{self.run_id}")
            socketio.emit('info', f"Starting new run: run_{self.run_id}")
        
        if not self.reset():
            socketio.emit('info', "All search missions have been completed!")
            return
        
        socketio.emit('info', "Algorithm is controlling the search process")
        self.b_time = time.time()
        self.l_time = self.b_time
        self.result_in_episode.append([self.episode, self.T_in_episode,
                                        self.b_time, self.v_x, self.v_y,
                                        0, 0])
        self.bf_algorithm_step()

    def save_result(self):
        dataNew = './' + user + '/AC-scenes' + str(self.scenes) + '.mat'
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
@socketio.on('upload')
def handle_upload(json):
    global main_instance
    main_instance = Main(map_file='uploaded_map.mat')
    emit('info', "Map uploaded successfully!")
    emit('enable_button', 'start')

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

@socketio.on('click')
def handle_click(json):
    main_instance.cf_board(json)

@socketio.on('start')
def handle_start():
    main_instance.bf_start()

@socketio.on('continue')
def handle_continue():
    global main_instance
    main_instance.bf_continue()

@socketio.on('execute')
def handle_execute():
    main_instance.bf_execute()

@socketio.on('next')
def handle_next():
    main_instance.bf_next()

@socketio.on('save_canvas_image')
def handle_save_canvas_image(json):
    image_data = json['image']
    image_data = image_data.split(',')[1]
    image_binary = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_binary))
    image_path = os.path.join(IMAGE_SAVE_DIR, f'canvas_{time.time()}.png')
    image.save(image_path)
    print(f"Canvas image saved to {image_path}")

if __name__ == "__main__":
    socketio.run(app,debug=True,allow_unsafe_werkzeug=True)
