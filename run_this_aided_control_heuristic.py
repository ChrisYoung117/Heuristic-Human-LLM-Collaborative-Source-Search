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
        
        # 用来存储完整的寻源过程数据
        self.process_data = {
            'steps': []
        }
        self.user_clicks = []
        self.trapped_state = []
        self.optional_areas = {}
        self.run_id = run_id if run_id else self._get_next_run_id()       
        # 输出目录
        self.output_dir = f'run_{self.run_id}/output'
        os.makedirs(self.output_dir, exist_ok=True)        
        self.completed_scenes = set()
        self.load_completed_scenes()     
        self.visit_counts = np.zeros((20, 20))

    def _get_next_run_id(self):
        existing_runs = []
        for dirname in os.listdir('.'):
            if dirname.startswith('run_') and os.path.isdir(dirname):
                try:
                    run_num = int(dirname.split('_')[1])
                    # 检查该run是否完成
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
        print(f"The completed scenes: {sorted(list(self.completed_scenes))}")
        print(f"The current run: run_{self.run_id}")

    def reset(self):
        self.episode += 1        
        if self.episode >= len(self.scenes_order[0]):
            return True
        
        while self.episode < len(self.scenes_order[0]):
            self.scenes = self.scenes_order[0, self.episode]
            if self.scenes not in self.completed_scenes:
                break
            print(f"Skip the completed scene {self.scenes}")
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
    #The main logic of the robot's automatic source search
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

    # randomly select a passable area
    def select_random_passable_area(self):
        current_x = int(math.floor(self.v_x / size))
        current_y = int(math.floor(self.v_y / size))
        passable_areas = []
        if not hasattr(self, 'selected_points'):
            self.selected_points = set()
        
        # find all the passable areas
        map_height, map_width = self.o_map.shape
        for x in range(min(20, map_height)):
            for y in range(min(20, map_width)):
                # check the conditions:
                # 1. the known area(unknow_map=1)
                # 2. the passable area(o_map=1)
                # 3. not in the nearby of the current position
                # 4. not in the selected points
                # 5. not in the forbidden area(o_map!=-1)
                if (self.unknow_map[x, y] == 1 and 
                    self.o_map[x, y] == 1 and 
                    not (abs(x - current_x) <= 1 and abs(y - current_y) <= 1) and
                    (x, y) not in self.selected_points and
                    self.o_map[x, y] != -1):
                    
                    # check if the area is reachable
                    myAstar = algorithm.AStar((current_x, current_y), (x, y), self.o_map, self.unknow_map)
                    if myAstar.run() == 1:  # if there is a path
                        passable_areas.append((x, y))
        
        if passable_areas:
            chosen_point = random.choice(passable_areas)
            self.selected_points.add(chosen_point)
            return chosen_point
        return None

    def select_area_by_particles(self):
        """根据粒子分布选择目标区域，动态平衡探索与利用"""
        current_x = int(math.floor(self.v_x / size))
        current_y = int(math.floor(self.v_y / size))

        self.visit_counts[current_x, current_y] += 1

        total_cells = 20 * 20
        explored_cells = np.sum(self.unknow_map)
        exploration_ratio = explored_cells / total_cells
        
        particle_weight = max(0.3, 0.9 - exploration_ratio * 0.6)  # 从0.9逐渐降至0.3
        exploration_weight = 1.0 - particle_weight  # 从0.1逐渐增至0.7
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
                        # 先测试是否能规划出路径
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
    
    def auto_escape_trap(self):
    
        best_area = self.select_area_by_particles()
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
            
            # 绘制路径
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
            
            # 记录被困状态
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
                        # 'scene_id': self.scenes,             # 当前场景ID
                        # 'episode': self.episode,             # 当前episode
                        'gas_concentration': self.detection if isinstance(self.detection, int) else self.detection.tolist()  # Handle detection type
                    }

                    step_id = f'{self.scenes:02}{self.T_in_episode:03}'
                    self.record_step(step_id, current_position, map_info)
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
                    self.auto_escape_trap()
                    if not self.auto_escape_trap():
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
    #获取用户画布点击坐标
    @staticmethod
    def obtain_area(e):
        x = int(math.floor(e['x'] / size / UNIT))
        y = int(math.floor(e['y'] / size / UNIT))
        return x, y

    #恢复机器人自动寻源逻辑
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
                                                0, 0])  # detection, Flag of Human(1) or Algorithm(0)
        main_instance.bf_algorithm_step()
        socketio.emit('enable_button', 'start')

    #开始人工控制寻源
    @staticmethod
    def bf_execute():
        if len(main_instance.draw_chosen_trajectory) > 0:
            main_instance.already_flag = 0
            main_instance.human_control_record.append(
                [main_instance.T_in_episode, main_instance.human_control_st, time.time(),
                 time.time() - main_instance.human_control_st, 1])  # 1:execute, 2:continue
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

    #进入下一张地图寻源
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
#处理上传地图文件
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

#监听前端点击事件
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

if __name__ == "__main__":
    socketio.run(app,debug=True,allow_unsafe_werkzeug=True)
