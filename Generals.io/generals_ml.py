import numpy as np
import random
import torch
import threading
import queue
from torch.autograd import Function
from torch.utils.data import Dataset
import torch.nn.functional as F
from torch import optim
import torch.nn as nn
import time
import matplotlib.pyplot as plt
from torch.autograd import Variable
import matplotlib.animation as animation

waitlist = queue.Queue(100)
ani_queue = queue.Queue(1000)

class Game():
	def __init__(self,map_size,moutain_density,castle_density,castle_force):
		self.map_size = map_size
		self.moutain_density = moutain_density
		self.castle_density = castle_density
		self.castle_force = castle_force
		self.__create_map()
		while self.grid[1].sum()<2:
			self.__create_map() 
	
	def __create_map(self):
		self.grid = np.zeros((5,self.map_size,self.map_size))
		random = np.random.rand(self.map_size,self.map_size)
		self.grid[0] = (random<self.moutain_density).astype(np.float32)
		self.grid[1] = ((random<self.moutain_density+self.castle_density)*(random>self.moutain_density)).astype(np.float32)
		player = np.random.rand(self.map_size,self.map_size)*self.grid[1]
		i1,j1 = np.unravel_index(player.argmax(), player.shape)
		player[i1,j1] = 0
		i2,j2 = np.unravel_index(player.argmax(), player.shape)
		self.grid[2,i1,j1] = 10
		self.grid[3,i2,j2] = 10
		self.grid[4] = self.grid[1]* self.castle_force-self.grid[2]-self.grid[3] 
		"""
		0:mountain
		1:castle
		2:player1's force
		3:player2's force
		4:castle's force
		"""

	def player1_maxPoss(self, action):
		policy = action
		if(self.player1_check_availability(policy)):
			position = [policy[1],policy[2]]
			outcome = self.player1_InvadeOutcome(position,self.player1_check_availability(policy)-1)
			return policy[0],policy[1],policy[2]
		else:
			raise TypeError

	def player1_check_availability(self,policy):
		if(policy[0]==0): #if goes up
			if(policy[1]<self.map_size-1 and self.grid[2][policy[1]+1][policy[2]]>1 and self.grid[0][policy[1]][policy[2]]==0):
				return self.grid[2][policy[1]+1][policy[2]] 
				#returns the amount of force
			else:
				return 0
		elif(policy[0]==1): #if goes down
			if(policy[1]>0 and self.grid[2][policy[1]-1][policy[2]]>1 and self.grid[0][policy[1]][policy[2]]==0):
				return self.grid[2][policy[1]-1][policy[2]]
			else:
				return 0
		elif(policy[0]==2): #if goes left
			if(policy[2]<self.map_size-1 and self.grid[2][policy[1]][policy[2]+1]>1 and self.grid[0][policy[1]][policy[2]]==0):
				return self.grid[2][policy[1]][policy[2]+1]
			else:
				return 0
		elif(policy[0]==3):	#if goes right
			if(policy[2]>0 and self.grid[2][policy[1]][policy[2]-1]>1 and self.grid[0][policy[1]][policy[2]]==0):
				return self.grid[2][policy[1]][policy[2]-1]
			else:
				return 0
	
	def player1_InvadeOutcome(self, position, force):
		if(self.grid[0][position[0]][position[1]]):
			return 0
		elif(self.grid[4][position[0]][position[1]]):# if it is a unoccupied castle
			if(force > self.grid[4][position[0]][position[1]]):#if it becomes player1's occupation
				self.grid[2][position[0]][position[1]] = force-self.grid[4][position[0]][position[1]]
				self.grid[4][position[0]][position[1]] = 0
				return 1
			else:#if it still is unoccupied castle
				self.grid[4][position[0]][position[1]] = self.grid[4][position[0]][position[1]]-force
				return 1
		elif(self.grid[2][position[0]][position[1]]):
			self.grid[2][position[0]][position[1]] = self.grid[2][position[0]][position[1]]+force
			return 1
		elif(self.grid[3][position[0]][position[1]]):
			if(force>self.grid[3][position[0]][position[1]]):
				self.grid[2][position[0]][position[1]] = force-self.grid[3][position[0]][position[1]]
				self.grid[3][position[0]][position[1]] = 0
				return 1
			elif(force<self.grid[3][position[0]][position[1]]):
				self.grid[3][position[0]][position[1]] = self.grid[3][position[0]][position[1]]-force
				return 1
			else:
				self.grid[2][position[0]][position[1]] = 0
				self.grid[3][position[0]][position[1]] = 0
				return 1
		else:
			self.grid[2][position[0]][position[1]] = force
			return 1	
	
	def player1_move(self, action):
		action = np.unravel_index(action, (4,8,8))
		action = [i[0] for i in action]
		try:
			direction, y, x = self.player1_maxPoss(action)
			if(direction==0):
				self.grid[2][y+1][x] = 1
			elif(direction==1):
				self.grid[2][y-1][x] = 1
			elif(direction==2):
				self.grid[2][y][x+1] = 1
			elif(direction==3):
				self.grid[2][y][x-1] = 1
			return self.grid
		except TypeError:
			return self.grid
		#deal with the force on the initial position
		

	"""
	move_map.size = (4,map_size,map_size)
	Save the possibility of moving to that position
	4:up/down/left/right
	"""
	#deal with player 2 movements
	def player2_maxPoss(self, sorted_map,i=-1):
		policy = action
		if(self.player2_check_availability(policy)):
			position = [policy[1],policy[2]]
			outcome = self.player2_InvadeOutcome(position,self.player2_check_availability(policy)-1)
			return policy[0],policy[1],policy[2]
		else:
			raise TypeError
			
	def check_legal_moves(self):
		player1_legal_moves=np.zeros((4,self.map_size,self.map_size))
		player2_legal_moves=np.zeros((4,self.map_size,self.map_size))
		for i in range(4):
			for j in range(self.map_size):
				for k in range(self.map_size):
					if(self.player1_check_availability([i,j,k])):
						player1_legal_moves[i,j,k] = 1
					if(self.player2_check_availability([i,j,k])):
						player2_legal_moves[i,j,k] = 1
						
		self.player1_legal_moves = player1_legal_moves
		self.player2_legal_moves = player2_legal_moves

	def player2_check_availability(self,policy):
		if(policy[0]==0): #if  goes up
			if(policy[1]<self.map_size-1 and self.grid[3][policy[1]+1][policy[2]]>1):
				return self.grid[3][policy[1]+1][policy[2]] 
				#returns the amount of force
			else:
				return 0
		elif(policy[0]==1): #if goes down
			if(policy[1]>0 and self.grid[3][policy[1]-1][policy[2]]>1):
				return self.grid[3][policy[1]-1][policy[2]]
			else:
				return 0
		elif(policy[0]==2): #if goes left
			if(policy[2]<self.map_size-1 and self.grid[3][policy[1]][policy[2]+1]>1):
				return self.grid[3][policy[1]][policy[2]+1]
			else:
				return 0
		elif(policy[0]==3):	#if goes right
			if(policy[2]>0 and self.grid[3][policy[1]][policy[2]-1]>1):
				return self.grid[3][policy[1]][policy[2]-1]
			else:
				return 0
	
	def player2_InvadeOutcome(self, position, force):
		if(self.grid[0][position[0]][position[1]]):
			return 0
		elif(self.grid[4][position[0]][position[1]]):# if it is a unoccupied castle
			if(force > self.grid[4][position[0]][position[1]]):#if it becomes player1's occupation
				self.grid[3][position[0]][position[1]] = force-self.grid[4][position[0]][position[1]]
				self.grid[4][position[0]][position[1]] = 0
				return 1
			else:#if it still is unoccupied castle
				self.grid[4][position[0]][position[1]] = self.grid[4][position[0]][position[1]]-force
				return 1
		elif(self.grid[3][position[0]][position[1]]):
			self.grid[3][position[0]][position[1]] = self.grid[3][position[0]][position[1]]+force
			return 1
		elif(self.grid[2][position[0]][position[1]]):
			if(force>self.grid[2][position[0]][position[1]]):
				self.grid[3][position[0]][position[1]] = force-self.grid[2][position[0]][position[1]]
				self.grid[2][position[0]][position[1]] = 0
				return 1
			elif(force<self.grid[2][position[0]][position[1]]):
				self.grid[2][position[0]][position[1]] = self.grid[2][position[0]][position[1]]-force
				return 1
			else:
				self.grid[3][position[0]][position[1]] = 0
				self.grid[2][position[0]][position[1]] = 0
				return 1
		else:
			self.grid[3][position[0]][position[1]] = force
			return 1

	def player2_move(self, action):
		action = np.unravel_index(action, (4,8,8))
		action = [i[0] for i in action]
		try:
			direction, y, x = self.player2_maxPoss(action)
			if(direction==0):
				self.grid[2][y+1][x] = 1
			elif(direction==1):
				self.grid[2][y-1][x] = 1
			elif(direction==2):
				self.grid[2][y][x+1] = 1
			elif(direction==3):
				self.grid[2][y][x-1] = 1
			return self.grid
		except TypeError:
			return self.grid
		#deal with the force on the initial position

	def update_force(self):
		self.grid[2] += (self.grid[2]>0)*0.1 + (self.grid[2]>0)*(self.grid[1]>0)*0.9
		self.grid[3] += (self.grid[3]>0)*0.1 + (self.grid[3]>0)*(self.grid[1]>0)*0.9

	def player1_map(self):
		grid = self.grid
		return grid.astype(np.float32)

	def player2_map(self):
		grid = self.grid
		grid[2] = self.grid[3]
		grid[3] = self.grid[2]
		return grid.astype(np.float32)

	def reward(self):
		force1 = np.sum(self.grid[2])
		force2 = np.sum(self.grid[3])
		forceDiff = force1 - force2
		castle1 = (self.grid[2]>0)*(self.grid[1]>0)*10
		castle2 = (self.grid[3]>0)*(self.grid[1]>0)*10
		castle1num = np.sum(castle1)
		castle2num = np.sum(castle2)
		castlenumdiff = castle1num - castle2num
		reward = forceDiff + castlenumdiff
		reward =np.tanh(reward/100)
		return reward

	def end_game(self):
		castle1 = (self.grid[2]>0)*(self.grid[1]>0)*1
		castle2 = (self.grid[3]>0)*(self.grid[1]>0)*1
		castle1num = np.sum(castle1)
		castle2num = np.sum(castle2)
		if castle1num == 0:
			return 2
		elif castle2num == 0:
			return 1
		else :
			return 0

	def generate_plot(self,grid):
		plot = np.tanh(np.copy(grid)/10)[2:5]
		plot = plot + grid[0].reshape(1,8,8).repeat(3,axis=0)*np.array([133,87,35]).reshape(3,1,1)/255
		return plot



class LambdaPad(nn.Module):

	def __init__(self, func, pad):
		super().__init__()
		self.func = func
		self.pad = pad
	def forward(self, x):
		return self.func(x, self.pad, mode='constant',value = 0)


class ResCell(nn.Module):
	def __init__(self,channels):
		super(ResCell, self).__init__()
		self.channels = channels
		self.conv = nn.Sequential(
			nn.BatchNorm2d(self.channels),
			nn.GELU(),            
			LambdaPad(F.pad,(1,1,1,1)),
			nn.Conv2d(self.channels, self.channels, (3, 3), stride=(1, 1)),
			nn.BatchNorm2d(self.channels),
			nn.GELU(),            
			LambdaPad(F.pad,(1,1,1,1)),
			nn.Conv2d(self.channels, self.channels, (3, 3), stride=(1, 1)),
		)

	def forward(self, x):
		output = 0.5*x + self.conv(x)
		return output


class ResLayer(nn.Module):
	def __init__(self,inc,outc):
		super(ResLayer, self).__init__()
		self.cell1 = ResCell(outc)
		self.cell2 = ResCell(outc)
		self.conv = nn.Sequential(
			nn.BatchNorm2d(inc),
			nn.GELU(),            
			LambdaPad(F.pad,(1,1,1,1)),
			nn.Conv2d(inc, outc, (3, 3), stride=(1, 1)),
			nn.BatchNorm2d(outc),
			nn.GELU(),            
			LambdaPad(F.pad,(1,1,1,1)),
			nn.Conv2d(outc, outc, (3, 3), stride=(1, 1)),
		)
	def forward(self, x):
		x = self.conv(x)
		x = self.cell1(x)
		output = self.cell2(x)
		return output


class TConv(nn.Module):
	def __init__(self,inc,outc):
		super(TConv, self).__init__()
		self.bnorm1 = nn.BatchNorm2d(inc)
		self.bnorm2 = nn.BatchNorm2d(inc)
		self.bnorm3 = nn.BatchNorm2d(inc)
		self.activation = nn.GELU()
		self.Tconv1 = nn.ConvTranspose2d(inc, inc, 3, stride=1)
		self.Tconv2 = nn.ConvTranspose2d(inc, inc, 3, stride=1)
		self.Tconv3 = nn.ConvTranspose2d(inc, outc, 3, stride=2)
		self.padding = nn.Sequential(
			LambdaPad(F.pad,(1,1,1,1)),
			)
	def forward(self, x):
		x = self.bnorm1(x)
		x = self.activation(x)
		x = self.padding(x)
		x = self.Tconv1(x)
		x = x[:,:,2:-2,2:-2]
		x = self.bnorm2(x)
		x = self.activation(x)
		x = self.padding(x)
		x = self.Tconv2(x)
		x = x[:,:,2:-2,2:-2]
		x = self.bnorm3(x)
		x = self.activation(x)
		x = self.padding(x)
		x = self.Tconv3(x)
		output = x[:,:,2:-3,2:-3]
		return output


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
			nn.Linear(576,1024,True),
			nn.Dropout(p = 0.4),
			nn.GELU(),
			nn.Linear(1024,1024,True),
			nn.Dropout(p = 0.4),
			nn.GELU(),
			nn.Linear(1024,1024,True),
			nn.Dropout(p = 0.4),
			nn.GELU(),
			nn.Linear(1024,1024,True),
			nn.Dropout(p = 0.4),
			nn.GELU(),
			nn.Linear(1024,512,True),
			nn.Dropout(p = 0.4),
			nn.GELU(),
			nn.Linear(512,256,True),
			nn.Dropout(p = 0.4),
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
		x = torch.cat((self.Flatten(grid),memory),1)

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
			nn.Linear(320,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),
			nn.Linear(512,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),
			nn.Linear(512,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),
			nn.Linear(512,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),
			nn.Linear(512,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),
			nn.Linear(512,512,True),
			nn.Dropout(p=0.4),
			nn.GELU(),         
			nn.Linear(512,1,True),
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
		return self.Actor(grid,memory), self.Value(grid)


class Training_arguments():
	def __init__(self):
		0.9,1e-4,0.1,8,1,0.001,0.001,0.01,0.005
		self.gamma = 0.9
		self.lr = 1e-6
		self.max_grad_norm = 0.1
		self.map_size = 8
		self.tau = 1
		self.entropy_coef = 0.001
		self.value_loss_coef = 0.001
		self.policy_loss_coef = 0.01
		self.off_tile_coef = 0.001
		self.reg_coef = 50


class Agent(threading.Thread):
	def __init__(self, model, args, waitlist, plot = True, print_ = False):
		threading.Thread.__init__(self)
		self.model = model
		# self.model.load_state_dict(torch.load('./checkpoint_DNN.pth'))
		self.model.eval()
		self.waitlist = waitlist
		self.plot = plot
		self.print_ = print_

	def new_episode(self):
		self.game = Game(self.args.map_size,0.1,0.2,10)
		self.game_ongoing = True
		self.counter = 0

	def evaluate_episode(self):
		global ani_queue
		logit1 = torch.zeros(1,256)
		logit2 = torch.zeros(1,256)
		old_logit1 = torch.zeros(256)
		old_logit2 = torch.zeros(256)
		while self.game_ongoing and self.counter < 200:

			logit1, _ = self.model(Variable(torch.tensor(self.game.player1_map().reshape(1,5,8,8))),logit1.clone().detach().view(1,-1))
			logit1 = logit1.contiguous().view(-1)
			prob1 = F.softmax(logit1,dim = 0)
			old_prob1 = prob1

			self.game.check_legal_moves()
			legal_moves1 = torch.tensor(self.game.player1_legal_moves)
			legal_moves1 = legal_moves1.contiguous().view(-1)
			prob1 = old_prob1 * Variable(legal_moves1.float())

			action1 = prob1.multinomial(1).clone().detach()
			old_logit1 = prob1.clone().detach()

			self.game_ongoing = (self.game.end_game() == 0)
			self.game.player1_move(action1.numpy())
			

			logit2, _ = self.model(Variable(torch.tensor(self.game.player2_map().reshape(1,5,8,8))),logit2.clone().detach().view(1,-1))
			logit2 = logit2.contiguous().view(-1)
			prob2 = F.softmax(logit2,dim = 0)
			old_prob2 = prob2

			self.game.check_legal_moves()
			legal_moves2 = torch.tensor(self.game.player2_legal_moves)
			legal_moves2 = legal_moves2.contiguous().view(-1)
			prob2 = old_prob2 * Variable(legal_moves2.float())

			action2 = prob2.multinomial(1).clone().detach()
			old_logit2 = prob2.clone().detach()

			self.game_ongoing = (self.game.end_game() == 0)
			self.game.player2_move(action2.numpy())

			self.game.update_force()
			if self.print_:
				print(self.game.reward()/100)
			if self.plot:
				ani_queue.put(self.game.generate_plot(np.copy(self.game.grid)))
			self.counter+=1

	def run(self):
		while True:
			self.new_episode()
			self.evaluate_episode()


args = Training_arguments()

fig = plt.figure(figsize=(8, 8))
agents = []
training_hist = []
waitlist.put("permission!")

def animate(i):
	plot = ani_queue.get()
	plt.cla()
	plot = np.swapaxes(plot,0,1)
	plot = np.swapaxes(plot,1,2)
	return plt.imshow(plot)

agent = Agent(ACmodel(),args,waitlist)

agent.start()

# agents[0].plot = True
# while True:
# 	try:
# 		anim = animation.FuncAnimation(fig, animate, frames =100, interval=10,save_count=60)
# 		plt.show()
# 	except:
# 		print("restart animation")
# anim.save("animation.gif",writer = 'pillow')

agent.join()

