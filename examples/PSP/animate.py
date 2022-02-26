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
from numpy.random import default_rng
from template_model import template_model
from template_mpc import template_mpc
from template_simulator import template_simulator


from matplotlib.animation import FuncAnimation, FFMpegWriter, ImageMagickWriter
from matplotlib.patches import Circle
from matplotlib.patches import Ellipse
from do_mpc.data import save_results, load_results

#results = load_results('./results/007_PSP_turn.pkl')
results = load_results('./results/pred_obs.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',2]
xd = results['mpc']['_x','x',1]
yd = results['mpc']['_x','x',3]
px = results['mpc']['_u','u',0]
py = results['mpc']['_u','u',1]

dyn = results['mpc']['_tvp','dyn_obs',0]
dyn_y = results['mpc']['_tvp','dyn_obs_y',0]
dyn_x = results['mpc']['_tvp','dyn_obs_x',0]
dyn_ry = results['mpc']['_tvp','dyn_obs_ry',0]
dyn_rx = results['mpc']['_tvp','dyn_obs_rx',0]
dyn_y_pred = results['mpc']['_tvp','dyn_obs_y_pred',0]
dyn_x_pred = results['mpc']['_tvp','dyn_obs_x_pred',0]
xsim = results['simulator']['_x','x',0]
ysim = results['simulator']['_x','x',2]


fig = plt.figure()
fig.set_dpi(100)
#fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-1, 7), ylim=(4, 11))

patch = plt.Circle((5, 8.5), 1, fc='y')
ax.plot(x,y)

patch2= plt.Circle((x[0],y[0]), 0.01)
patch3= plt.Circle((-px[0]+x[0],-py[0]+y[0]), 0.01)

circle1 = Circle((5, 8.5), 1, alpha=0.5)
circle2 = Circle((8, 9), 0.7, alpha=0.5)
goal = Circle((5, 10), 0.1, color="green", label="goal")
#circle1 = Circle((0.5, 9.25), 0.5, alpha=0.5)
#circle2 = Circle((1.8, 9.25), 0.5, alpha=0.5)
#goal = Circle((1.5, 8.5), 0.1, color="red", alpha=0.5)
#ax.add_artist(circle1)
#ax.add_artist(circle2)
ax.add_artist(goal)

patchp = Ellipse(xy=(dyn_x_pred[0], dyn_y_pred[0]), width=0.7*2,
                        height=2*0.7, 
                        edgecolor='r', fc='None', lw=2, label="Prediction")

#ax.add_patch(patchp)
#print(np.sqrt(((x[1]-8)/0.7*1.2)**2 + ((y[1]-dyn[0])/0.7*1.4)**2)-1)

patch = Circle((dyn_x[0], dyn_y[0]), 0.7, fc='y', alpha=0.2, label="obstacle")

def init():
    #ax.add_patch(patchp)
    return patchp, patch

def animate(i): 
    
    x_s = x[i]
    y_s = y[i]
    patch2= plt.Circle((x_s,y_s), 0.08)
    x_f = px[i]+ x[i]
    y_f = py[i]+ y[i]
    patch3= plt.Circle((x_f,y_f), 0.08, fc='black')
    ax.add_artist(patch2)
    ax.add_artist(patch3)

    x_obs = dyn_x[i]
    y_obs = dyn_y[i]
    #patch = plt.Circle((x_obs, y_obs), 0.7, fc='y', alpha=0.9)
    patch.set_center((x_obs, y_obs)) #just moving the plot and nnot over plotting
    x_obsp = dyn_x_pred[i]
    y_obsp = dyn_y_pred[i]
    patchp.set_center((x_obsp, y_obsp))
    patchp.height = 2*dyn_ry[i]*0.7
    patchp.width = 2*dyn_rx[i]*0.7

    
    ax.add_artist(patch)

    #ax.add_patch(patchp)

    
    return patchp, patch,
    
#plt.scatter(dyn_x,dyn_y, color='y')
    
#plt.legend()


ax.title.set_text('n_horz = 3 for Obstacle Prediction')
anim = FuncAnimation(fig, animate, init_func=init, interval=500)



plt.show()
