import pygame
import numpy as np
from collections import deque
import random
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
pygame.init()

black = (0, 0, 0)
white = (255, 255, 255)
gray = (220, 220, 220)
darkgray = (128, 128, 128)
ddarkgray = (108, 108, 108, 100)
green = (84, 173, 50)
ggreen = (0, 100, 0)
red = (219, 59, 38)


class Actor(nn.Module):
    def __init__(self):
        super(Actor, self).__init__()
        # self.Res1 = ResLayer(5,64)
        # self.Res2 = ResLayer(64,128)
        # self.Res3 = ResLayer(128,128)
        # self.Res4 = ResLayer(128,128)
        # self.Pool1 = nn.Sequential(
        # 	LambdaPad(F.pad,(1,1,1,1)),
        # 	nn.Conv2d(128, 128, (3, 3), stride=(2, 2)),
        # 	)
        self.DNN = nn.Sequential(
            nn.Flatten(),
            nn.Linear(576, 1024, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(1024, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 256, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
        )
        self.Flatten = nn.Flatten()

    def forward(self, grid, memory):
        grid = torch.tanh(grid/100)
        # grid = self.Res1(grid)
        # grid = self.Res2(grid)
        # grid = self.Pool1(grid)
        # grid = self.Res3(grid)
        # grid = self.Res4(grid)
        x = torch.cat((self.Flatten(grid), memory), 1)

        output = self.DNN(x)
        return output


class Value(nn.Module):
    def __init__(self):
        super(Value, self).__init__()
        # self.Res1 = ResLayer(5,64)
        # self.Res2 = ResLayer(64,128)
        self.Flatten = nn.Flatten()
        # self.Pool1 = nn.Sequential(
        # 	LambdaPad(F.pad,(1,1,1,1)),
        # 	nn.Conv2d(64, 64, (3, 3), stride=(2, 2)),
        # 	)
        self.Dnn = nn.Sequential(
            nn.Linear(320, 1024, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(1024, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 2048, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(2048, 512, True),
            nn.Dropout(p=0.4),
            nn.GELU(),
            nn.Linear(512, 1, True),
        )

    def forward(self, grid):
        grid = torch.tanh(grid/100)
        # grid = self.Res1(grid)
        # grid = self.Pool1(grid)
        # grid = self.Res2(grid)
        output = self.Flatten(grid)
        output = self.Dnn(output)
        output = 10*torch.tanh(output)
        return output


class ACmodel(nn.Module):
    def __init__(self):
        super(ACmodel, self).__init__()
        self.Actor = Actor()
        self.Value = Value()

    def forward(self, grid, memory):
        return self.Actor(grid, memory), self.Value(grid)


class play_game():
    def __init__(self, width, height, map):
        self.width = width
        self.height = height
        self.map = np.array(map)
        '''
        map:
        0   mountains
        1   towers
        2   force 1 land
        3   force 2 land
        4   castle force

        '''
        self.map_size = len(map[1])
        self.gameExit = False
        self.gameEnd = False
        self.gameDisplay = pygame.display.set_mode((width, height))
        pygame.display.set_caption('Generals.io')
        self.clock = pygame.time.Clock()
        self.block_size = width / self.map_size
        self.trajectory_map = deque([])
        self.images()
        PATH = "checkpoint_DNN.pth"
        self.model = ACmodel()
        self.model.load_state_dict(torch.load(
            PATH, map_location=torch.device('cpu')))
        self.logit = torch.zeros(1, 256)
        self.old_logit = self.logit

    def enemy_move(self):
        self.logit, _ = self.model(torch.tensor(
            self.map.reshape(1, 5, 8, 8).astype(np.float32)), self.logit.view(1, -1))
        self.logit = self.logit.contiguous().view(-1)
        prob = F.softmax(1000*self.logit, dim=0)
        old_prob = prob

        legal_moves = self.check_legal_moves()

        legal_moves = torch.tensor(legal_moves)
        legal_moves = legal_moves.contiguous().view(-1)
        prob = old_prob * Variable(legal_moves.float())

        action = prob.multinomial(1).clone().detach()
        self.old_logit = prob.clone().detach()
        return np.unravel_index(action.numpy(), (4, 8, 8))

    def enemy_map_move(self):
        move = self.enemy_move()
        move = [move[0][0], move[1][0], move[2][0]]
        destination = [move[1], move[2]]

        # find current force
        if move[0] == 0:
            curr_force = self.map[3][destination[0] + 1][destination[1]]-1
        elif move[0] == 1:
            curr_force = self.map[3][destination[0] - 1][destination[1]]-1
        elif move[0] == 2:
            curr_force = self.map[3][destination[0]][destination[1] + 1]-1
        elif move[0] == 3:
            curr_force = self.map[3][destination[0]][destination[1] - 1]-1

        def oriland(move):
            if move[0] == 0:
                self.map[3][destination[0] + 1][destination[1]] = 1
            elif move[0] == 1:
                self.map[3][destination[0] - 1][destination[1]] = 1
            elif move[0] == 2:
                self.map[3][destination[0]][destination[1] + 1] = 1
            elif move[0] == 3:
                self.map[3][destination[0]][destination[1] - 1] = 1

        if self.check_availability(move) and not self.map[0][destination[0]][destination[1]]:

            # if enemy steps on his land/tower
            if self.map[3][destination[0]][destination[1]]:
                self.map[3][destination[0]][destination[1]] += curr_force
                oriland(move)

            # if enemy steps on player's land/tower
            elif self.map[2][destination[0]][destination[1]]:
                if curr_force > self.map[2][destination[0]][destination[1]]:
                    self.map[3][destination[0]][destination[1]] = curr_force - \
                        self.map[2][destination[0]][destination[1]]
                    self.map[2][destination[0]][destination[1]] = 0
                    oriland(move)
                elif curr_force < self.map[2][destination[0]][destination[1]]:
                    self.map[2][destination[0]][destination[1]] -= curr_force
                    oriland(move)

            # if enemy step on tower
            elif self.map[1][destination[0]][destination[1]]:
                # double check if enemy steps on vacant tower
                if self.map[4][destination[0]][destination[1]]:
                    if curr_force > self.map[4][destination[0]][destination[1]]:
                        self.map[3][destination[0]][destination[1]] = curr_force - \
                            self.map[4][destination[0]][destination[1]]
                        self.map[4][destination[0]][destination[1]] = 0
                        oriland(move)
                    elif curr_force <= self.map[4][destination[0]][destination[1]]:
                        self.map[4][destination[0]
                                    ][destination[1]] -= curr_force
                        oriland(move)
            # enemy steps on vacant land
            else:
                self.map[3][destination[0]][destination[1]] = curr_force
                oriland(move)
        else:
            return

    def check_legal_moves(self):
        legal_moves = np.zeros((4, self.map_size, self.map_size))
        for i in range(4):
            for j in range(self.map_size):
                for k in range(self.map_size):
                    if self.check_availability([i, j, k]):
                        legal_moves[i, j, k] = 1
        return legal_moves

    def check_availability(self, policy):
        if(policy[0] == 0):  # if  goes up
            if(policy[1] < self.map_size-1 and self.map[3][policy[1]+1][policy[2]] > 1):
                return self.map[3][policy[1]+1][policy[2]]
                # returns the amount of force
            else:
                return 0
        elif(policy[0] == 1):  # if goes down
            if(policy[1] > 0 and self.map[3][policy[1]-1][policy[2]] > 1):
                return self.map[3][policy[1]-1][policy[2]]
            else:
                return 0
        elif(policy[0] == 2):  # if goes left
            if(policy[2] < self.map_size-1 and self.map[3][policy[1]][policy[2]+1] > 1):
                return self.map[3][policy[1]][policy[2]+1]
            else:
                return 0
        elif(policy[0] == 3):  # if goes right
            if(policy[2] > 0 and self.map[3][policy[1]][policy[2]-1] > 1):
                return self.map[3][policy[1]][policy[2]-1]
            else:
                return 0

    def images(self):
        self.towerImg = pygame.image.load('./images/tower.png')
        self.towerImg = pygame.transform.smoothscale(
            self.towerImg, (int(self.block_size * 0.85), int(self.block_size * 0.85)))
        self.towerImg_green = pygame.image.load('./images/tower-green.png')
        self.towerImg_green = pygame.transform.smoothscale(
            self.towerImg_green, (int(self.block_size * 0.85), int(self.block_size * 0.85)))
        self.towerImg_red = pygame.image.load('./images/tower-red.png')
        self.towerImg_red = pygame.transform.smoothscale(
            self.towerImg_red, (int(self.block_size * 0.85), int(self.block_size * 0.85)))
        self.mountainImg = pygame.image.load('./images/mountain.png')
        self.mountainImg = pygame.transform.smoothscale(
            self.mountainImg, (int(self.block_size * 0.85), int(self.block_size * 0.85)))

    def static_map(self):
        self.gameDisplay.fill(gray)
        self.line_width = int(self.block_size//30)
        x = 0

        while x <= self.width:
            x += self.block_size
            pygame.draw.line(self.gameDisplay, black,
                             (x, 0), (x, self.height), self.line_width)
            pygame.draw.line(self.gameDisplay, black,
                             (0, x), (self.width, x), self.line_width)

    def trajectory(self):
        al = 0.5 * self.block_size  # arrow length
        pd = 0.25 * self.block_size  # paddding
        ct = 0.5 * self.block_size  # center
        i = 1
        while i < len(self.trajectory_map)-1 and self.map[2][self.trajectory_map[0][1]][self.trajectory_map[0][0]] > 1:
            x1, y1 = self.trajectory_map[i]
            x2, y2 = self.trajectory_map[i + 1]

            if x2 - x1 == 1:
                x1 *= self.block_size
                y1 *= self.block_size
                pygame.draw.polygon(self.gameDisplay, (0, 0, 0), ((x1+pd, y1+ct-0.25), (x1+pd, y1+ct+0.25), (x1+pd+al-3, y1+ct+0.25),
                                                                  (x1+pd+al-3, y1+ct+3), (x1+pd+al, y1+ct), (x1+pd+al-3, y1+ct-3), (x1+pd+al-3, y1+ct-0.25), (x1+pd, y1+ct-0.25)))
            elif y1 - y2 == 1:
                x1 *= self.block_size
                y1 *= self.block_size
                pygame.draw.polygon(self.gameDisplay, (0, 0, 0), ((x1+ct, y1+pd), (x1+ct-3, y1+pd+3), (x1+ct-0.25, y1+pd+3),
                                                                  (x1+ct-0.25, y1+pd+al), (x1+ct+0.25, y1+pd+al), (x1+ct+0.25, y1+pd+3), (x1+ct+3, y1+pd+3), (x1+ct, y1+pd)))
            elif x1 - x2 == 1:
                x1 *= self.block_size
                y1 *= self.block_size
                pygame.draw.polygon(self.gameDisplay, (0, 0, 0), ((x1+pd, y1+ct), (x1+pd+3, y1+ct+3), (x1+pd+3, y1+ct+0.25),
                                                                  (x1+pd+al, y1+ct+0.25), (x1+pd+al, y1+ct-0.25), (x1+pd+3, y1+ct-0.25), (x1+pd+3, y1+ct-3), (x1+pd, y1+ct)))
            elif y2 - y1 == 1:
                x1 *= self.block_size
                y1 *= self.block_size
                pygame.draw.polygon(self.gameDisplay, (0, 0, 0), ((x1+ct-0.25, y1+pd), (x1+ct+0.25, y1+pd), (x1+ct+0.25, y1+pd+al-3),
                                                                  (x1+ct+3, y1+pd+al-3), (x1+ct, y1+pd+al), (x1+ct-3, y1+pd+al-3), (x1+ct-0.25, y1+pd+al-3), (x1+ct-0.25, y1+pd)))
            i += 1

    def add_traject(self, x, y):
        self.trajectory_map.append((x, y))

    def tower(self):
        for (y, x), element in np.ndenumerate(self.map[1]):
            if element:
                if self.map[2][y][x]:
                    self.gameDisplay.blit(
                        self.towerImg_green, ((x + 0.075)*self.block_size, (y + 0.075)*self.block_size))
                elif self.map[3][y][x]:
                    self.gameDisplay.blit(
                        self.towerImg_red, ((x + 0.075)*self.block_size, (y + 0.075)*self.block_size))
                else:
                    self.gameDisplay.blit(
                        self.towerImg, ((x + 0.075)*self.block_size, (y + 0.075)*self.block_size))

    def towerRect(self):
        for (y, x), element in np.ndenumerate(self.map[1]):
            if element:
                pygame.draw.rect(self.gameDisplay, darkgray, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size+(
                    y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 0)

    def mountain(self):
        for (y, x), element in np.ndenumerate(self.map[0]):
            # print(self.map[1])
            if element:
                self.gameDisplay.blit(
                    self.mountainImg, ((x + 0.075)*self.block_size, (y + 0.075)*self.block_size))

    def mountainRect(self):
        for (y, x), element in np.ndenumerate(self.map[0]):
            if element:
                pygame.draw.rect(self.gameDisplay, darkgray, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size+(
                    y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 0)

    def force(self, number, font):
        forceTextSurface = font.render(number, True, white)
        return forceTextSurface, forceTextSurface.get_rect()

    def force_display(self):
        text = pygame.font.SysFont('poppinslight', 13)
        arr = self.map[2] + self.map[3] + self.map[4]
        for (y, x), element in np.ndenumerate(arr):
            if(arr[y][x]):

                TextSurf, TextRect = self.force(str(int(arr[y][x])), text)
                TextRect.center = ((x+0.5)*self.block_size,
                                   (y+0.5)*self.block_size)
                self.gameDisplay.blit(TextSurf, TextRect)

    def land_display(self):
        arr = self.map[2] + self.map[3] + self.map[4]
        for (y, x), element in np.ndenumerate(arr):
            if arr[y][x]:
                if self.map[2][y][x]:
                    pygame.draw.rect(self.gameDisplay, green, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size +
                                                               (y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 0)
                elif self.map[3][y][x]:
                    pygame.draw.rect(self.gameDisplay, red, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size +
                                                             (y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 0)

    def map_move(self):
        if len(self.trajectory_map) >= 2:
            if self.map[2][self.trajectory_map[0][1]][self.trajectory_map[0][0]] > 1:
                curr_force = self.map[2][self.trajectory_map[0]
                                         [1]][self.trajectory_map[0][0]] - 1
                # if I step on a tower
                if self.map[1][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                    # if the tower belongs to me
                    if self.map[2][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                        self.map[2][self.trajectory_map[1][1]][self.trajectory_map[1][0]
                                                               ] = curr_force + self.map[2][self.trajectory_map[1][1]][self.trajectory_map[1][0]]
                        self.map[2][self.trajectory_map[0]
                                    [1]][self.trajectory_map[0][0]] = 1
                        self.trajectory_map.popleft()
                    # else if the tower belongs to enemy
                    elif self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:

                        if curr_force > self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                            self.map[2][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = curr_force - self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]
                            self.map[3][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = 0
                            self.map[2][self.trajectory_map[0]
                                        [1]][self.trajectory_map[0][0]] = 1
                            self.trajectory_map.popleft()
                        elif curr_force <= self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                            self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]
                                                                   ] = self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]] - curr_force
                            self.map[2][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = 0
                            self.map[2][self.trajectory_map[0]
                                        [1]][self.trajectory_map[0][0]] = 1
                            self.trajectory_map = deque([])
                    # else if the tower belongs to no one
                    elif self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:

                        if curr_force > self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                            self.map[2][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = curr_force - self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]]
                            self.map[4][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = 0
                            self.map[2][self.trajectory_map[0]
                                        [1]][self.trajectory_map[0][0]] = 1
                            self.trajectory_map.popleft()
                        elif curr_force <= self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                            self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]
                                                                   ] = self.map[4][self.trajectory_map[1][1]][self.trajectory_map[1][0]] - curr_force
                            self.map[2][self.trajectory_map[1]
                                        [1]][self.trajectory_map[1][0]] = 0
                            self.map[2][self.trajectory_map[0]
                                        [1]][self.trajectory_map[0][0]] = 1
                            self.trajectory_map = deque([])
                # else if the land belongs to enemy
                elif self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                    if curr_force > self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]:
                        self.map[3][self.trajectory_map[1]
                                    [1]][self.trajectory_map[1][0]] = 0
                        self.map[2][self.trajectory_map[1]
                                    [1]][self.trajectory_map[1][0]] = curr_force - self.map[3][self.trajectory_map[1][1]][self.trajectory_map[1][0]]
                        self.map[2][self.trajectory_map[0]
                                    [1]][self.trajectory_map[0][0]] = 1
                        self.trajectory_map.popleft()
                    else:
                        self.map[3][self.trajectory_map[1]
                                    [1]][self.trajectory_map[1][0]] -= curr_force
                        self.map[2][self.trajectory_map[0]
                                    [1]][self.trajectory_map[0][0]] = 1
                        self.trajectory_map = deque([])
                else:
                    self.map[2][self.trajectory_map[1][1]][self.trajectory_map[1][0]
                                                           ] += curr_force
                    self.map[2][self.trajectory_map[0]
                                [1]][self.trajectory_map[0][0]] = 1
                    self.trajectory_map.popleft()
            else:
                self.trajectory_map = deque(
                    [])

    def start(self):
        self.game_loop()

    def player_position(self, x, y):
        if(self.map[2][y][x]):
            pygame.draw.rect(self.gameDisplay, white, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size+(
                y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 1*self.line_width)
        else:
            pygame.draw.rect(self.gameDisplay, green, (x*self.block_size+(x > 0)*self.line_width, y*self.block_size+(
                y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 2*self.line_width)

    def player_position_side(self, x, y):
        pygame.draw.rect(self.gameDisplay, ddarkgray, ((x-1)*self.block_size+(x > 1)*self.line_width, y*self.block_size+(
            y > 0)*self.line_width, self.block_size - (x > 1)*self.line_width, self.block_size - (y > 0)*self.line_width), 0)
        pygame.draw.rect(self.gameDisplay, ddarkgray, ((x+1)*self.block_size+(x > 0)*self.line_width, y*self.block_size+(
            y > 0)*self.line_width, self.block_size - (x > 0)*self.line_width, self.block_size - (y > 0)*self.line_width), 0)
        pygame.draw.rect(self.gameDisplay, ddarkgray, ((x)*self.block_size+(x > 0)*self.line_width, (y-1)*self.block_size+(
            y > 1)*self.line_width, self.block_size - (x > 0)*self.line_width, self.block_size - (y > 1)*self.line_width), 0)
        pygame.draw.rect(self.gameDisplay, ddarkgray, ((x)*self.block_size+(x > 0)*self.line_width, (y+1)*self.block_size+(
            y > 0)*self.line_width, self.block_size-(x > 0)*self.line_width, self.block_size-(y > 0)*self.line_width), 0)

    def tower_update(self):
        self.map[2] += (self.map[2] > 0) * (self.map[1] > 0) * 1
        self.map[3] += (self.map[3] > 0) * (self.map[1] > 0) * 1

    def land_update(self):
        self.map[2] += (self.map[2] > 0) * (self.map[1] == 0) * 1
        self.map[3] += (self.map[3] > 0) * (self.map[1] == 0) * 1

    def game_loop(self):
        choice_list = []
        for (i, j), element in np.ndenumerate(self.map[1]):
            if element:
                choice_list.append((i, j))
        choice = random.randrange(0, len(choice_list) - 1)
        choiceEnemy = random.randrange(0, len(choice_list) - 1)

        while choiceEnemy == choice:
            choiceEnemy = random.randrange(0, len(choice_list) - 1)

        self.posx = choice_list[choice][1]
        self.posy = choice_list[choice][0]
        self.map[2][self.posy][self.posx] = 10
        self.map[4][self.posy][self.posx] = 0
        self.map[3][choice_list[choiceEnemy][0]
                    ][choice_list[choiceEnemy][1]] = 10
        self.map[4][choice_list[choiceEnemy][0]
                    ][choice_list[choiceEnemy][1]] = 0
        # add initial position to trajectory map
        self.add_traject(self.posx, self.posy)

        i = 0
        j = 0
        while not self.gameExit:
            if np.all(self.map[2] == 0):
                self.gameExit = True
            elif np.all(self.map[3] == 0):
                self.gameExit = True
            elif np.all(self.map[4] == 0):
                self.gameExit = True

            self.static_map()
            self.towerRect()
            self.mountainRect()
            self.land_display()

            if i % 30 == 0:
                self.map_move()
                self.enemy_map_move()
                self.tower_update()
                j += 1
                if j % 10 == 0:
                    self.land_update()

            # self.player_position_side(self.posx, self.posy)
            self.mountain()

            for event in pygame.event.get():

                # quit when the window closes
                if event.type == pygame.QUIT:
                    pygame.quit()
                    quit()

                if event.type == pygame.KEYDOWN:

                    if event.key == pygame.K_LEFT and self.posx != 0 and self.map[0][self.posy][self.posx-1] == 0:
                        self.posx -= 1
                    elif event.key == pygame.K_RIGHT and self.posx != self.map_size-1 and self.map[0][self.posy][self.posx+1] == 0:
                        self.posx += 1
                    elif event.key == pygame.K_UP and self.posy != 0 and self.map[0][self.posy-1][self.posx] == 0:
                        self.posy -= 1
                    elif event.key == pygame.K_DOWN and self.posy != self.map_size-1 and self.map[0][self.posy+1][self.posx] == 0:
                        self.posy += 1
                    elif event.key == pygame.K_q:
                        self.trajectory_map = deque([self.trajectory_map[0]])
                        self.posx = self.trajectory_map[0][0]
                        self.posy = self.trajectory_map[0][1]
                        continue
                    else:
                        continue

                    self.add_traject(self.posx, self.posy)
            self.tower()
            self.trajectory()
            self.force_display()
            self.player_position(self.posx, self.posy)

            pygame.display.update()
            self.clock.tick(60)
            i += 1


# creating random map
SIZE = 8
map1 = np.random.rand(SIZE, SIZE)  # castle
map2 = np.zeros((SIZE, SIZE))  # mountain
map3 = np.zeros((SIZE, SIZE))  # player1 force
map4 = np.zeros((SIZE, SIZE))  # player2 force
map5 = np.zeros((SIZE, SIZE))  # tower force


for (x, y), element in np.ndenumerate(map1):
    if element > 0.1 and element < 0.9:
        map1[x][y] = 0
    elif element > 0.9:
        map2[x][y] = 1
        map5[x][y] = random.randrange(5, 20)
        map1[x][y] = 0
    else:
        map1[x][y] = 1

mapt = [map1, map2, map3, map4, map5]

game1 = play_game(800, 800, mapt)
game1.start()
