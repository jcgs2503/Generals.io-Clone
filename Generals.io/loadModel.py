import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random

PATH = "checkpoint_DNN.pth"
model = torch.load(PATH, map_location=torch.device('cpu'))

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
logit1 = torch.seros(1,256)

while true:
    logit1, _ = model(torch.tensor(mapt.reshape(1, 5, 8, 8), logit.view(1, -1)))
    logit1 = logit1.contiguous().view(-1)
    prob1 = F.softmax(logit1, dim=0)
    old_prob1 = prob1

    game.check_legal_moves()
    legal_moves1 = torch.tensor(self.game.player1_legal_moves)
    legal_moves1 = legal_moves1.contiguous().view(-1)
    prob1 = old_prob1 * Variable(legal_moves1.float())

    action = prob1.multinomial(1).clone().detach().numpy()
    old_logit1 = prob1.clone().detach()
    action = np.unravel_index(action,(4,8,8))
print()
