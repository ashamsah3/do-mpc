import numpy as np
from discret_risk_map import *
from eucl2disc import *
from GridWorld import GridWorld
from ValueIteration import ValueIteration
from scipy.interpolate import UnivariateSpline
import time


xfull=[]
yfull=[]
x=0.5
y=0.5
xfull.append(x)
yfull.append(y)


problem = GridWorld('data/world004.csv', reward={0: 0.0, 1: 5.0, 2: -5.0, 3: np.NaN}, random_rate=0.2)

for i in range(3,14):
	
	ind = int(i)
	CVaR, CVaR_space , x_space, y_space = CVaR_map(0.95, 1, ind)
	#tic = time.perf_counter()
	solver = ValueIteration(problem.reward_function, problem.transition_model, CVaR, gamma=0.99)
	solver.train()
	#problem.visualize_value_policy(policy=solver.policy, values=solver.values)

	s = eucl2disc(x,y,8,6,1)
	g = 23
	#toc = time.perf_counter()
	#print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
	
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
	
	fig2 = plt.figure()
	ax1 = fig2.add_subplot(1,1,1)
	colormesh = ax1.pcolormesh(x_space+0.5,y_space+0.5,CVaR_space, alpha=0.3)
	#fig2.colorbar(colormesh, ax=ax1)
	ax1.scatter(xfull,yfull)
	border = Rectangle((0,0), 8, 6, fc = "None", ec="black" )
	ax1.add_patch(border)
	wall = Rectangle((3,0), 1, 1, fc = "gray", ec="black" )
	wall2 = Rectangle((3,3), 1, 3, fc = "gray", ec="black" )
	goal = Rectangle((7,3), 1, 1, fc = "green", ec="black" )
	ax1.add_patch(wall)
	ax1.add_patch(wall2)
	ax1.add_patch(goal)
	#xs = np.linspace(xfull[0], 8, 1000)
	#path = UnivariateSpline(xfull, yfull)
	#ax1.scatter(xfull,path(xfull))
	plt.show()

	x = xfull[1]
	y=yfull[1]
	xfull=[]
	yfull=[]
	xfull.append(x)
	yfull.append(y)
	
	

#problem.random_start_policy(policy=solver.policy, start_pos=(1, 1), n=1000)

#problem.visualize_value_policy(policy=solver.policy, values=solver.values)
'''
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

'''
#fig2 = plt.figure()
#ax1 = fig2.add_subplot(2,1,1)
#ax1.scatter(xfull,yfull)
plt.show()

