import numpy as np
from math import floor

UNIT = 25
X_min = 0
Y_min = 0
X_max = 20
Y_max = 20
size = 1
step_size = 1

class Maze:
    def __init__(self, params):
        self.params = params
        self.canvas_width = (params['X_max'] + 11) * params['UNIT']
        self.canvas_height = (params['Y_max'] + 7) * params['UNIT']
        self.explored_area = np.zeros((params['X_max'], params['Y_max']), dtype=bool)
        self.obstacle_info = np.zeros((params['X_max'], params['Y_max']), dtype=int)  # Store obstacle information
        self.lastpoint = None

    def draw_reset(self, v_x, v_y, s_x, s_y, o_map, x, y):
        data = []

        # 初始化所有方格为黑色（未知状态）
        for i in range(self.params['X_max']):
            for j in range(self.params['Y_max']):
                data.append({'type': 'rect', 'x': i * self.params['UNIT'], 'y': j * self.params['UNIT'],
                             'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': '#000000'})

        # 重置已探索区域
        self.explored_area = np.zeros(
            (self.params['X_max'] - self.params['X_min'] + 1, self.params['Y_max'] - self.params['Y_min'] + 1),
            dtype=bool)

        # 只照亮机器人的起始位置及其周围区域
        for i in range(-1, 2):
            for j in range(-1, 2):
                vx = int(v_x) + i
                vy = int(v_y) + j
                if (0 <= vx < self.params['X_max']) and (0 <= vy < self.params['Y_max']):
                    self.explored_area[vx, vy] = True
                    color = '#FFFFFF'  # 默认为白色（已探索区域）
                    if o_map[vx, vy] == 0:  # 假设0代表障碍物
                        color = '#808080'  # 障碍物颜色
                        self.obstacle_info[vx, vy] = 1  # 标记障碍物
                    data.append({'type': 'rect', 'x': vx * self.params['UNIT'], 'y': vy * self.params['UNIT'],
                                 'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': color})

        # 绘制粒子浓度分布
        data.extend(self.draw_particles(x, y))

        # 绘制机器人当前位置
        data.append(
            {'type': 'oval', 'x': v_x * self.params['UNIT'] - 6, 'y': v_y * self.params['UNIT'] - 6, 'radius': 6,
             'fill': 'blue'})
        return data

    def draw_update(self, v_x, v_y, o_map, x, y):
        data = []

        # 清除画布上的旧粒子图和地图环境
        data.append({'type': 'clear'})

        # 绘制所有方格为黑色（未知状态）
        for i in range(self.params['X_max']):
            for j in range(self.params['Y_max']):
                if not self.explored_area[i, j]:
                    data.append({'type': 'rect', 'x': i * self.params['UNIT'], 'y': j * self.params['UNIT'],
                                 'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': '#000000'})

        # 更新已探索区域的颜色，包括障碍物信息
        for i in range(self.params['X_max']):
            for j in range(self.params['Y_max']):
                if self.explored_area[i, j]:
                    color = '#FFFFFF'  # 默认为白色（已探索区域）
                    if self.obstacle_info[i, j] == 1:  # 检查是否是障碍物
                        color = '#808080'  # 障碍物颜色
                    data.append({'type': 'rect', 'x': i * self.params['UNIT'], 'y': j * self.params['UNIT'],
                                 'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': color})

        # 揭示机器人当前所在的位置，并更新已探索状态
        for i in range(-1, 2):
            for j in range(-1, 2):
                vx = int(v_x) + i
                vy = int(v_y) + j
                if (0 <= vx < self.params['X_max']) and (0 <= vy < self.params['Y_max']):
                    self.explored_area[vx, vy] = True
                    color = '#FFFFFF'  # 默认为白色（已探索区域）
                    if o_map[vx, vy] == 0:  # 假设0代表障碍物
                        color = '#808080'  # 障碍物颜色
                        self.obstacle_info[vx, vy] = 1  # 标记障碍物
                    data.append({'type': 'rect', 'x': vx * self.params['UNIT'], 'y': vy * self.params['UNIT'],
                                 'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': color})

        # 绘制粒子浓度分布
        data.extend(self.draw_particles(x, y))

        # 绘制机器人当前位置
        data.append(
            {'type': 'oval', 'x': v_x * self.params['UNIT'] - 6, 'y': v_y * self.params['UNIT'] - 6, 'radius': 6,
             'outline': 'blue'})

        # 更新 lastpoint
        self.lastpoint = [v_x, v_y]

        # 如果 lastpoint 已经被设置，绘制轨迹线
        if self.lastpoint is not None:
            data.append({'type': 'line', 'x1': self.lastpoint[0] * self.params['UNIT'],
                         'y1': self.lastpoint[1] * self.params['UNIT'], 'x2': v_x * self.params['UNIT'],
                         'y2': v_y * self.params['UNIT']})
        return data

    def draw_particles(self, x, y):
        particle_size = 2
        data = []
        for i in range(len(x)):
            data.append({'type': 'oval', 'x': x[i] * self.params['UNIT'] - particle_size, 'y': y[i] * self.params['UNIT'] - particle_size, 'radius': particle_size, 'outline': 'green'})
        return data

    def update_unknown_area(self, v_x, v_y):
        vx = int(floor(v_x / size))
        vy = int(floor(v_y / size))
        data = []

        for i in range(-1, 2):
            for j in range(-1, 2):
                x = v_x + i * step_size
                y = v_y + j * step_size
                if (x <= self.params['X_max']) and (x >= self.params['X_min']) and \
                        (y <= self.params['Y_max']) and (y >= self.params['Y_min']):
                    data.append({'type': 'delete', 'id': (vx + i * step_size) * 20 + vy + j * step_size})
        return data

    def draw_detection(self, z, v_x, v_y):
        data = []
        if z > 0:
            data.append({'type': 'rect', 'x': self.params['UNIT'] * v_x - 4, 'y': self.params['UNIT'] * v_y - 4,
                         'width': 8, 'height': 8, 'fill': 'magenta'})
        else:
            data.append({'type': 'oval', 'x': self.params['UNIT'] * v_x - 2, 'y': self.params['UNIT'] * v_y - 2,
                         'radius': 2, 'fill': 'magenta'})
        return data

    def draw_fb(self, x, y):
        i = int(floor(x / size))
        j = int(floor(y / size))
        data = [{'type': 'rect', 'x': i * self.params['UNIT'], 'y': j * self.params['UNIT'],
                 'width': self.params['UNIT'], 'height': self.params['UNIT'], 'fill': '#B4B4B4'}]
        return data

    def draw_action(self, v_x, v_y, want_action, real_action):
        data = []
        x1 = 10
        x2 = 6
        if real_action == 0:
            data.append(
                {'type': 'polygon', 'points': [(v_x + step_size) * self.params['UNIT'] + x1, v_y * self.params['UNIT'],
                                               (v_x + step_size) * self.params['UNIT'], v_y * self.params['UNIT'] + x2,
                                               (v_x + step_size) * self.params['UNIT'], v_y * self.params['UNIT'] - x2],
                 'fill': 'red'})
        elif real_action == 1:
            data.append(
                {'type': 'polygon', 'points': [v_x * self.params['UNIT'], (v_y + step_size) * self.params['UNIT'] + x1,
                                               v_x * self.params['UNIT'] + x2, (v_y + step_size) * self.params['UNIT'],
                                               v_x * self.params['UNIT'] - x2, (v_y + step_size) * self.params['UNIT']],
                 'fill': 'red'})
        elif real_action == 2:
            data.append(
                {'type': 'polygon', 'points': [(v_x - step_size) * self.params['UNIT'] - x1, v_y * self.params['UNIT'],
                                               (v_x - step_size) * self.params['UNIT'], v_y * self.params['UNIT'] + x2,
                                               (v_x - step_size) * self.params['UNIT'], v_y * self.params['UNIT'] - x2],
                 'fill': 'red'})
        elif real_action == 3:
            data.append(
                {'type': 'polygon', 'points': [v_x * self.params['UNIT'], (v_y - step_size) * self.params['UNIT'] - x1,
                                               v_x * self.params['UNIT'] + x2, (v_y - step_size) * self.params['UNIT'],
                                               v_x * self.params['UNIT'] - x2, (v_y - step_size) * self.params['UNIT']],
                 'fill': 'red'})
        if want_action != real_action:
            if want_action == 0:
                data.append({'type': 'polygon',
                             'points': [(v_x + step_size) * self.params['UNIT'] + x1, v_y * self.params['UNIT'],
                                        (v_x + step_size) * self.params['UNIT'], v_y * self.params['UNIT'] + x2,
                                        (v_x + step_size) * self.params['UNIT'], v_y * self.params['UNIT'] - x2],
                             'fill': 'blue'})
            elif want_action == 1:
                data.append({'type': 'polygon',
                             'points': [v_x * self.params['UNIT'], (v_y + step_size) * self.params['UNIT'] + x1,
                                        v_x * self.params['UNIT'] + x2, (v_y + step_size) * self.params['UNIT'],
                                        v_x * self.params['UNIT'] - x2, (v_y + step_size) * self.params['UNIT']],
                             'fill': 'blue'})
            elif want_action == 2:
                data.append({'type': 'polygon',
                             'points': [(v_x - step_size) * self.params['UNIT'] - x1, v_y * self.params['UNIT'],
                                        (v_x - step_size) * self.params['UNIT'], v_y * self.params['UNIT'] + x2,
                                        (v_x - step_size) * self.params['UNIT'], v_y * self.params['UNIT'] - x2],
                             'fill': 'blue'})
            elif want_action == 3:
                data.append({'type': 'polygon',
                             'points': [v_x * self.params['UNIT'], (v_y - step_size) * self.params['UNIT'] - x1,
                                        v_x * self.params['UNIT'] + x2, (v_y - step_size) * self.params['UNIT'],
                                        v_x * self.params['UNIT'] - x2, (v_y - step_size) * self.params['UNIT']],
                             'fill': 'blue'})
        return data