import numpy as np
from discret_risk_map import *
from eucl2disc import *
from GridWorld import GridWorld
from ValueIteration import ValueIteration

problem = GridWorld('data/world004.csv', reward={0: 0.0, 1: 5.0, 2: -5.0, 3: np.NaN}, random_rate=0.2)

CVaR = CVaR_map(0.95, 2,2+3)
solver = ValueIteration(problem.reward_function, problem.transition_model, CVaR, gamma=0.99)
solver.train()

problem.visualize_value_policy(policy=solver.policy, values=solver.values)
#problem.random_start_policy(policy=solver.policy, start_pos=(1, 1), n=1000)

xfull=[]
yfull=[]
x=1.5
y=2.5
xfull.append(x)
yfull.append(y)
s = eucl2disc(x,y,8,6,1)
g = 23

while s != g:
	action = solver.policy[s]
	#symbol = ['^', '>', 'v', '<']
	if action == 0:
		yfull.append(y+1)
		xfull.append(x)
		y=y+1
	elif action == 1:
		xfull.append(x+1)
		yfull.append(y)
		x=x+1
	elif action == 2:
		yfull.append(y-1)
		xfull.append(x)
		y=y-1
	elif action == 3:
		xfull.append(x-1)
		yfull.append(y)
		x=x-1
	s = eucl2disc(x,y,8,6,1)
	#print(s)


fig2 = plt.figure()
ax1 = fig2.add_subplot(2,1,1)
ax1.scatter(xfull,yfull)
plt.show()

