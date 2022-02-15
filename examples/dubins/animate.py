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

#results = load_results('./results/007_PSP_turn.pkl')
results = load_results('./results/003_turn_case_study.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',2]
xd = results['mpc']['_x','x',1]
yd = results['mpc']['_x','x',3]
px = results['mpc']['_u','u',0]
py = results['mpc']['_u','u',1]

dyn = results['mpc']['_tvp','dyn_obs',0]
xsim = results['simulator']['_x','x',0]
ysim = results['simulator']['_x','x',2]


fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-1, 10), ylim=(4, 12))
patch = plt.Circle((5, 8.5), 1, fc='y')

ax.plot(x,y)

patch2= plt.Circle((x[0],y[0]), 0.1)
patch3= plt.Circle((-px[0]+x[0],-py[0]+y[0]), 0.1)

#circle1 = Circle((5, 8.5), 1, alpha=0.5)
#circle2 = Circle((8, 9), 0.7, alpha=0.5)
#goal = Circle((9, 5), 0.1, color="red", alpha=0.5)
circle1 = Circle((0.5, 9.25), 0.5, alpha=0.5)
circle2 = Circle((1.8, 9.25), 0.5, alpha=0.5)
goal = Circle((1.5, 8.5), 0.1, color="red", alpha=0.5)
ax.add_artist(circle1)
ax.add_artist(circle2)
ax.add_artist(goal)

def animate(i): 
	x_obs = 8 + 0*dyn[i]
	y_obs = dyn[i]
	patch = plt.Circle((x_obs, y_obs), 0.7, fc='y', alpha=0.2)
	x_s = x[i]
	y_s = y[i]
	patch2= plt.Circle((x_s,y_s), 0.1)
	x_f = px[i]+ x[i]
	y_f = py[i]+ y[i]
	patch3= plt.Circle((x_f,y_f), 0.1, fc='red')
	ax.add_artist(patch)
	ax.add_artist(patch2)
	ax.add_artist(patch3)
    




anim = FuncAnimation(fig, animate, interval=500)
'''
def init():
    patch.center = (5, 8.5)
    ax.add_patch(patch)
    return patch,



def animate(i):
    x_obs, y_obs = patch.center
    x_obs = 5 + 0*dyn[i]
    y_obs = dyn[i]
    patch.center = (x_obs, y_obs)
    return patch,


def init1():
    patch2.center = ((x[0],y[0]))
    ax.add_patch(patch2)
    return patch2,



def animate1(c):
    x_s, y_s = patch2.center
    x_s = x[c]
    y_s = y[c]
    patch2.center = (x_s, y_s)
    return patch2,


anim1 = animation.FuncAnimation(fig, animate1, 
                               init_func=init1,                            
                               interval=100,
                               blit=True)


anim = animation.FuncAnimation(fig, animate, 
                               init_func=init,                             
                               interval=100,
                               blit=True)

'''



plt.show()