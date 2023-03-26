import random
import csv
import math

# 定义导弹类
class Missile:
    def __init__(self, type, x, y, height, speed, status, group, number, launch_round, distance, is_launched=False):
        self.type = type
        self.x = x
        self.y = y
        self.height = height
        self.speed = speed
        self.status = status
        self.group = group
        self.number = number
        self.launch_round = launch_round
        self.distance = distance
        self.intercept_round = None
        self.is_launched = is_launched  # 记录导弹是否已经发射

    def get_status(self):
        return self.status

    def set_status(self, status):
        self.status = status

    def get_position(self):
        return self.x, self.y

    def set_position(self, x, y):
        self.x = x
        self.y = y

    def get_group(self):
        return self.group

    def get_launch_round(self):
        return self.launch_round

    def get_height(self):
        return self.height

    def get_distance(self):
        return self.distance

    def set_intercept_round(self, intercept_round):
        self.intercept_round = intercept_round

    def get_intercept_round(self):
        return self.intercept_round

    def launch(self, launch_round):
        # 生成随机初始坐标
        if self.x is None or self.y is None:
            self.x, self.y = generate_random_coordinates()
        # 更新导弹的发射回合和初始位置
        self.launch_round = launch_round
        self.is_launched = True

    def get_state(self):
        return str(self.number) + '_' + str(self.x) + '_' + str(self.y) + '_' + str(self.speed) + '_' + str(
            self.height) + '_' + str(self.launch_round)
    
    def clone(self):
        return Missile(self.type, self.x, self.y, self.height, self.speed, self.status, self.group, self.number, self.launch_round, self.distance, self.is_launched)

    @classmethod
    def from_csv_row(cls, row):
        type, x, y, height, speed, status, group, number, launch_round, distance = row
        return cls(type, float(x), float(y), int(height), float(speed), int(status), int(group), int(number),
                   int(launch_round), float(distance))

# 生成随机坐标
def generate_random_coordinates():
    return float(random.randint(21, 22)), float(random.randint(21, 22))

# 生成随机高度
def generate_random_height():
    return random.choice([1, 2, 3])

# 生成随机导弹分组和出场轮次
def generate_random_groups(total_missiles, num_groups):
    group_sizes = [0] * num_groups
    remaining_missiles = total_missiles
    for i in range(num_groups - 1):
        group_size = random.randrange(0, remaining_missiles // 10 + 1) * 10
        group_sizes[i] = group_size
        remaining_missiles -= group_size
    group_sizes[-1] = remaining_missiles
    random.shuffle(group_sizes)
    while True:
        groups = []
        launch_rounds = []
        for i in range(num_groups):
            launch_round = random.randint(0, 9)
            for j in range(group_sizes[i]):
                groups.append(i)
                launch_rounds.append(launch_round)
        if 0 in launch_rounds:
            break
    zipped = list(zip(groups, launch_rounds))
    random.shuffle(zipped)
    return zip(*zipped)

# 生成导弹列表
def generate_missile_list(num_missiles, num_groups, file_index):
    random.seed(file_index)  # 根据file_index设置随机种子
    missile_list = []
    groups, launch_rounds = generate_random_groups(num_missiles, num_groups)
    for i in range(num_missiles):
        type = 0
        x, y = generate_random_coordinates()
        height = generate_random_height()
        speed = 0.026
        status = 0
        group = groups[i]
        number = i
        launch_round = launch_rounds[i]
        distance = ((x ** 2) + (y ** 2))**0.5
        missile = Missile(type, x, y, height, speed, status, group, number, launch_round, distance)
        missile_list.append(missile)
    return missile_list

# 将导弹信息存入CSV文件
def save_missile_list_to_csv(missile_list, filename, file_index):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Type', 'X', 'Y', 'Height', 'Speed', 'Status', 'Group', 'Number', 'Launch_Round', 'Distance'])
        for missile in missile_list:
            writer.writerow([missile.type, missile.x, missile.y, missile.height, missile.speed,
                             missile.status, missile.group, missile.number, missile.launch_round, missile.distance])

num_missiles = 120
num_files = 1
num_groups = 5

for i in range(num_files):
    missile_list = generate_missile_list(num_missiles, num_groups,0)
    filename = f'missiles_{i+1}.csv'