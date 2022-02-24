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
results = load_results('./results/003_s2s_rand.pkl')

x = results['mpc']['_x','x',0]
y = results['mpc']['_x','x',2]
xd = results['mpc']['_x','x',1]
yd = results['mpc']['_x','x',3]
px = results['mpc']['_u','u',0]
py = results['mpc']['_u','u',1]

dyn = results['mpc']['_tvp','dyn_obs',0]
dyn_y = results['mpc']['_tvp','dyn_obs',0]
dyn_x = results['mpc']['_tvp','dyn_obs_x',0]
xsim = results['simulator']['_x','x',0]
ysim = results['simulator']['_x','x',2]


fig = plt.figure()
fig.set_dpi(100)
fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(4, 12), ylim=(4, 12))
patch = plt.Circle((5, 8.5), 1, fc='y')
ax.plot(x,y)

patch2= plt.Circle((x[0],y[0]), 0.1)
patch3= plt.Circle((-px[0]+x[0],-py[0]+y[0]), 0.1)

circle1 = Circle((5, 8.5), 1, alpha=0.5)
circle2 = Circle((8, 9), 0.7, alpha=0.5)
goal = Circle((10, 5), 0.1, color="red", alpha=0.5)
#circle1 = Circle((0.5, 9.25), 0.5, alpha=0.5)
#circle2 = Circle((1.8, 9.25), 0.5, alpha=0.5)
#goal = Circle((1.5, 8.5), 0.1, color="red", alpha=0.5)
#ax.add_artist(circle1)
#ax.add_artist(circle2)
ax.add_artist(goal)

patchp = Ellipse(xy=(dyn_x[0], dyn[0]), width=2*0.7*1.2,
                        height=2*1.6*0.7, 
                        edgecolor='y', fc='None', lw=2)

#ax.add_patch(patchp)
#print(np.sqrt(((x[1]-8)/0.7*1.2)**2 + ((y[1]-dyn[0])/0.7*1.4)**2)-1)

patch = plt.Circle((dyn_x[0], dyn[0]), 0.7, fc='y', alpha=0.9)

def init():
    #ax.add_patch(patchp)
    return patchp, patch

def animate(i): 
    
    x_s = x[i]
    y_s = y[i]
    patch2= plt.Circle((x_s,y_s), 0.1)
    x_f = px[i]+ x[i]
    y_f = py[i]+ y[i]
    patch3= plt.Circle((x_f,y_f), 0.1, fc='red')
    ax.add_artist(patch2)
    ax.add_artist(patch3)

    x_obs = dyn_x[i]
    y_obs = dyn_y[i]
    #patch = plt.Circle((x_obs, y_obs), 0.7, fc='y', alpha=0.9)
    patch.set_center((x_obs, y_obs)) #just moving the plot and nnot over plotting
    x_obsp = 8 + dyn[i]*0
    y_obsp = dyn[i]-0.3
    

    
    #patchp.set_center((x_obsp, y_obsp))
    
    ax.add_artist(patch)

    #ax.add_patch(patchp)

    
    return patchp, patch,
    
    
   
    




anim = FuncAnimation(fig, animate, init_func=init, interval=150)



plt.show()