import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from casadi import *
from casadi.tools import *
import pdb
import sys
import time
sys.path.append('../../')
import do_mpc

from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib.patches import Circle
from do_mpc.data import save_results, load_results


#results = load_results('./results/006_PSP_turn.pkl')
results = load_results('./results/003_stuck.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',2]
xd = results['mpc']['_x','x',1]
yd = results['mpc']['_x','x',3]
px = results['mpc']['_u','u',0]
py = results['mpc']['_u','u',1]

dyn = results['mpc']['_tvp','dyn_obs',0]
dth = results['mpc']['_aux','delta_th']
h = results['mpc']['_aux','obstacle_distance']

#hk1 = results['mpc']['_aux','hk1']
#hk1_n = results['mpc']['_aux','hk1_n']
xsim = results['simulator']['_x','x',0]
ysim = results['simulator']['_x','x',2]

stance = results['mpc']['_tvp','stance',0]

fig = plt.figure()
ax1 = fig.add_subplot(2,1,1)
ax2 = fig.add_subplot(2,1,2)
#ax3 = fig.add_subplot(3,1,3)
ax1.axis('equal')
ax1.plot(x,y)
ax1.scatter(-px+x,-py+y)
ax1.scatter(x,y)
ax1.scatter(((dyn-dyn)+1)*8,dyn)
circle1 = Circle((5, 8.5), 1, alpha=0.5)  
circle2 = Circle((8, 9), 0.7, alpha=0.5) # (2.5 , 9.25) >> turn_case_study, (2, 9.25) >>  001_turn_case_study
goal = Circle((9, 5), 0.1, color="red", alpha=0.5)
ax1.add_artist(circle1)
ax1.add_artist(circle2)
ax1.add_artist(goal)
plt.title('LIP with PSP Constraints')
#ax2.plot((1-0.5)*hk1 - hk1_n)
ax2.plot(x,h)
'''
ax2.scatter(px+x,px*0)
ax2.scatter(x,xd)
ax3.scatter(py,y-y)
ax3.scatter(y-y,yd)
'''

print(dth)
print(np.cos(dth))
#print(px)
#ax3.scatter(py+y,py*0)
plt.show()