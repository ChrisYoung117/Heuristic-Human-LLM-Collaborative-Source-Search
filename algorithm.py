import numpy as np
import scipy.stats as stats
from pylab import *
# from run_this import *
from sklearn.cluster import DBSCAN
from collections import Counter
import math
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


def diff(vx, vy, sx, sy, a, q, v, d, t):
    """
    gas particle diffusion function
    vx/vy is the robot location,sx/sy is the source location \
    """
    lam = sqrt((d * t) / (1 + (v ** 2) * t / (4 * d)))
    f = sqrt((vx - sx)**2 + (vy - sy)**2)
    s = exp(-(sx - vx)*v/(2*d))
    k = exp(-f / lam)
    r = a * q / f
    r = r * k * s
    return r
class ParticleFilter():

    def __init__(self):
        # initialize the particles
        self.number = particle_number
        self.x = np.random.uniform(X_min, X_max, self.number)
        self.y = np.random.uniform(Y_min, Y_max, self.number)
        self.weight = np.full(self.number, 1/self.number)

    def update(self, r_x, r_y, detection):
        lamd = diff(r_x, r_y, self.x, self.y, a, Q, V, D, t)
        probablity = stats.poisson.pmf(detection, lamd)
        self.weight *= probablity
        self.weight /= sum(self.weight)

    def resample(self):
        new_particle_x = np.zeros(self.number)
        new_particle_y = np.zeros(self.number)

        cumsum_weight = np.cumsum(self.weight)
        for i in range(self.number):
            rand_number = np.random.rand()
            target = next(n for n in cumsum_weight if n > rand_number)
            index = np.argwhere(cumsum_weight == target)

            new_particle_x[i] = self.x[index[0]]
            new_particle_y[i] = self.y[index[0]]

        self.x = new_particle_x
        self.y = new_particle_y
        self.weight = np.full(self.number, 1 / self.number)

    def mcmcStep(self, r_x, r_y, detection):

        for i in range(self.number):
            x_candidate = np.random.normal(loc=self.x[i], scale=0.2)
            y_candidate = np.random.normal(loc=self.y[i], scale=0.2)
            if (x_candidate < X_max) & (x_candidate > X_min) & \
               (y_candidate < Y_max) & (y_candidate > Y_min):
                mean1 = diff(r_x, r_y, x_candidate, y_candidate, a, Q, V, D, t)
                mean2 = diff(r_x, r_y, self.x[i], self.y[i], a, Q, V, D, t)
                prob1 = stats.poisson.pmf(detection, mean1)
                prob2 = stats.poisson.pmf(detection, mean2)
                alpha = min(1, prob1 / prob2)
                if np.random.rand() <= alpha:
                    self.x[i] = x_candidate
                    self.y[i] = y_candidate

    def reset(self):
        self.x = np.random.uniform(X_min, X_max, self.number)
        self.y = np.random.uniform(Y_min, Y_max, self.number)
        self.weight = np.full(self.number, 1 / self.number)

    def save(self):
        X = np.vstack((self.x, self.y)).T
        np.savetxt('data.txt', X)




def if_forbidden(v_x, v_y, o_map):
    count = [1, 1, 1, 1]  # right up left down 0: passable, 1: unpassable
    vx = int(floor(v_x/size))
    vy = int(floor(v_y/size))

    if v_x + step_size < X_max:
        if vx+1 < len(o_map) and vy < len(o_map[vx+1]):  # 添加边界检查
            if o_map[vx+1, vy] == 1:
                count[0] = 0

    if v_y + step_size < Y_max:
        if vx < len(o_map) and vy+1 < len(o_map[vx]):  # 添加边界检查
            if o_map[vx, vy+1] == 1:
                count[1] = 0

    if v_x - step_size > X_min:
        if vx-1 >= 0 and vy < len(o_map[vx-1]):  # 添加边界检查
            if o_map[vx-1, vy] == 1:
                count[2] = 0

    if v_y - step_size > Y_min:
        if vx < len(o_map) and vy-1 >= 0:  # 添加边界检查
            if o_map[vx, vy-1] == 1:
                count[3] = 0

    forbidden_flag = 0

    if sum(count) == 3:
        forbidden_flag = 1
    elif sum(count) == 2:
        h1 = [1, 0, 1, 0]
        h2 = [0, 1, 0, 1]

        if count != h1 and count != h2:
            xx = vx + step_size * (count[2] - count[0])
            yy = vy + step_size * (count[3] - count[1])
            if vx+1 < len(o_map) and vy+1 < len(o_map[vx+1]) and o_map[xx, yy] == 1:  # 添加边界检查
                forbidden_flag = 1

    return forbidden_flag


def infotaixs(r_x, r_y, pf):
    max_d = 21
    threshold = 0.99

    pro = pf.weight
    I = [0, 0, 0, 0]

    P = [0 for _ in range(max_d)]
    for de in range(max_d):
        lamd = diff(r_x + step_size, r_y, pf.x, pf.y, a, Q, V, D, t)
        probablity = stats.poisson.pmf(de, lamd)
        zwp = probablity * pro
        zwp_norm = zwp / sum(zwp)
        P[de] = (-sum(probablity * zwp_norm * np.log(zwp_norm + (zwp_norm == 0)))
                 + sum(pro * np.log(pro + (pro == 0))))
        I[0] -= P[de]
        if sum(P) > threshold * pf.number:
            break

    P = [0 for _ in range(max_d)]
    for de in range(max_d):
        lamd = diff(r_x, r_y + step_size, pf.x, pf.y, a, Q, V, D, t)
        probablity = stats.poisson.pmf(de, lamd)
        zwp = probablity * pro
        zwp_norm = zwp / sum(zwp)
        P[de] = (-sum(probablity * zwp_norm * np.log(zwp_norm + (zwp_norm == 0)))
                 + sum(pro * np.log(pro + (pro == 0))))
        I[1] -= P[de]
        if sum(P) > threshold * pf.number:
            break

    P = [0 for _ in range(max_d)]
    for de in range(max_d):
        lamd = diff(r_x - step_size, r_y, pf.x, pf.y, a, Q, V, D, t)
        probablity = stats.poisson.pmf(de, lamd)
        zwp = probablity * pro
        zwp_norm = zwp / sum(zwp)
        P[de] = (-sum(probablity * zwp_norm * np.log(zwp_norm + (zwp_norm == 0)))
                 + sum(pro * np.log(pro + (pro == 0))))
        I[2] -= P[de]
        if sum(P) > threshold * pf.number:
            break

    P = [0 for _ in range(max_d)]
    for de in range(max_d):
        lamd = diff(r_x, r_y - step_size, pf.x, pf.y, a, Q, V, D, t)
        probablity = stats.poisson.pmf(de, lamd)
        zwp = probablity * pro
        zwp_norm = zwp / sum(zwp)
        P[de] = (-sum(probablity * zwp_norm * np.log(zwp_norm + (zwp_norm == 0)))
                 + sum(pro * np.log(pro + (pro == 0))))
        I[3] -= P[de]
        if sum(P) > threshold * pf.number:
            break

    return I



def I2II(v_x, v_y, o_map, I):

    vx = int(floor(v_x / size))
    vy = int(floor(v_y / size))

    II = [500, 500, 500, 500]
    if v_x + step_size < X_max:
        if o_map[vx+1, vy] == 1:
            II[0] = I[0]

    if v_y + step_size < Y_max:
        if o_map[vx, vy+1] == 1:
            II[1] = I[1]

    if v_x - step_size > X_min:
        if o_map[vx-1, vy] == 1:
            II[2] = I[2]

    if v_y - step_size > Y_min:
        if o_map[vx, vy-1] == 1:
            II[3] = I[3]

    return II




def if_manual(T, trajectory):
    manual_flag = 0

    if T > 10:

        tra = trajectory[-10:]
        m = Counter(tra).most_common(1)
        if m[0][1] >= 5:
            manual_flag = 1


    return manual_flag

def manual_control(v_x, v_y, action, o_map):
    t_flag = 1
    manual_flag = 1

    while t_flag:
        command = input('请输入指令w,s,a,d:')
        if command == 'd':
            t_vx = v_x + step_size
            t_vy = v_y
            action = np.vstack((action, np.array([1, 0]).reshape(1, 2)))
        elif command == 's':
            t_vx = v_x
            t_vy = v_y + step_size
            action = np.vstack((action, np.array([0, 1]).reshape(1, 2)))
        elif command == 'a':
            t_vx = v_x - step_size
            t_vy = v_y
            action = np.vstack((action, np.array([-1, 0]).reshape(1, 2)))
        elif command == 'w':
            t_vx = v_x
            t_vy = v_y - step_size
            action = np.vstack((action, np.array([0, -1]).reshape(1, 2)))
        elif command == 'q':
            manual_flag = 0
            t_vx = v_x
            t_vy = v_y
            break
        else:
            continue

        if (t_vx <= X_max) and (t_vx >= X_min) and (t_vy <= Y_max) and (t_vy >= Y_min):
            vx = int(floor(t_vx / size))
            vy = int(floor(t_vy / size))
            if o_map[vx, vy] == 1:
                t_flag = 0

    return t_vx, t_vy, action, manual_flag


class Node(object):
    def __init__(self, pos):
        self.pos = pos
        self.father = None
        self.gvalue = 0
        self.fvalue = 0

    def compute_fx(self, enode, father):
        if father == None:
            print('未设置当前节点的父节点！')

        gx_father = father.gvalue
        # 采用欧式距离计算父节点到当前节点的距离
        gx_f2n = math.sqrt((father.pos[0] - self.pos[0]) ** 2 + (father.pos[1] - self.pos[1]) ** 2)
        gvalue = gx_f2n + gx_father

        hx_n2enode = math.sqrt((self.pos[0] - enode.pos[0]) ** 2 + (self.pos[1] - enode.pos[1]) ** 2)
        fvalue = gvalue + hx_n2enode
        return gvalue, fvalue

    def set_fx(self, enode, father):
        self.gvalue, self.fvalue = self.compute_fx(enode, father)
        self.father = father

    def update_fx(self, enode, father):
        gvalue, fvalue = self.compute_fx(enode, father)
        if fvalue < self.fvalue:
            self.gvalue, self.fvalue = gvalue, fvalue
            self.father = father


class AStar(object):
    def __init__(self, pos_sn, pos_en, o_map, unknown_area):
        self.mapsize = 20  # 表示地图的投影大小，并非屏幕上的地图像素大小
        self.openlist, self.closelist = [], []
        self.map = o_map
        self.unknown_area = unknown_area
        self.snode = Node(pos_sn)  # 用于存储路径规划的起始节点
        self.enode = Node(pos_en)  # 用于存储路径规划的目标节点
        self.cnode = self.snode  # 用于存储当前搜索到的节点

    def run(self):
        self.openlist.append(self.snode)
        while (len(self.openlist) > 0):
            # 查找openlist中fx最小的节点
            fxlist = list(map(lambda x: x.fvalue, self.openlist))
            index_min = fxlist.index(min(fxlist))
            self.cnode = self.openlist[index_min]
            del self.openlist[index_min]
            self.closelist.append(self.cnode)

            # 扩展当前fx最小的节点，并进入下一次循环搜索
            self.extend(self.cnode)
            # 如果openlist列表为空，或者当前搜索节点为目标节点，则跳出循环
            if len(self.openlist) == 0 or self.cnode.pos == self.enode.pos:
                break

        if self.cnode.pos == self.enode.pos:
            self.enode.father = self.cnode.father
            return 1
        else:
            return -1

    def get_minroute(self):
        minroute = []
        current_node = self.enode

        while (True):
            minroute.append(current_node.pos)
            current_node = current_node.father
            if current_node.pos == self.snode.pos:
                break

        minroute.append(self.snode.pos)
        minroute.reverse()
        return minroute

    def extend(self, cnode):
        nodes_neighbor = self.get_neighbor(cnode)
        for node in nodes_neighbor:
            # 判断节点node是否在closelist和blocklist中，因为closelist和blocklist中元素均为Node类，所以要用map函数转换为坐标集合
            if node.pos in list(map(lambda x: x.pos, self.closelist)):
                continue
            else:
                if node.pos in list(map(lambda x: x.pos, self.openlist)):
                    node.update_fx(self.enode, cnode)
                else:
                    node.set_fx(self.enode, cnode)
                    self.openlist.append(node)

    def setBlock(self, blocklist):
        '''
        获取地图中的障碍物节点，并存入self.blocklist列表中
        注意：self.blocklist列表中存储的是障碍物坐标，不是Node类
        :param blocklist:
        :return:
        '''
        self.blocklist.extend(blocklist)

    def get_neighbor(self, cnode):
        offsets = [(0, 1), (-1, 0), (1, 0), (0, -1)]
        nodes_neighbor = []
        x, y = cnode.pos[0], cnode.pos[1]
        for os in offsets:
            x_new, y_new = x + os[0], y + os[1]
            pos_new = (x_new, y_new)
            # 判断是否在地图范围内,超出范围跳过
            if 0 <= x_new < self.mapsize and 0 <= y_new < self.mapsize:
                if self.unknown_area[x_new, y_new] == 1 and self.map[x_new, y_new] == 1:
                    nodes_neighbor.append(Node(pos_new))

        return nodes_neighbor


def obtain_cluster(pfx, pfy):
    X = np.vstack((pfx, pfy)).T
    y_pred = DBSCAN(eps=0.4, min_samples=8).fit_predict(X)
    y_pred += 1
    count = np.bincount(y_pred)
    valid_cluster = count[count > 2000 * 0.05]
    vc_num = len(valid_cluster)

    if count[0] > 2000 * 0.05:
        vc_num -= 1

    if vc_num > 0:
        x_sum = 0
        y_sum = 0
        index = np.delete(count, 0).argmax() + 1
        num = np.delete(count, 0).max()
        for i in range(2000):
            if y_pred[i] == index:
                x_sum += X[i, 0]
                y_sum += X[i, 1]
        mean_x = x_sum / num
        mean_y = y_sum / num

    else:
        mean_x = 99
        mean_y = 99

    return mean_x, mean_y

