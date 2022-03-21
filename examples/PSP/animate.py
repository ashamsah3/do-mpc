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
from matplotlib.patches import Arrow
from do_mpc.data import save_results, load_results

#results = load_results('./results/007_PSP_turn.pkl')
results = load_results('./results/heading.pkl')

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

gamma = results['mpc']['_aux','gamma']
hn_aug = results['mpc']['_aux','hn_aug']

heading = results['mpc']['_aux','heading']
heading_g=[]

for i in range(1,len(heading)-1):
    print(i)
    heading_g.append(sum(heading[0:i])[0])

print(heading_g)

dyn_y2 = results['mpc']['_tvp','dyn_obs_y2',0]
dyn_x2 = results['mpc']['_tvp','dyn_obs_x2',0]
dyn_ry2 = results['mpc']['_tvp','dyn_obs_ry2',0]
dyn_rx2 = results['mpc']['_tvp','dyn_obs_rx2',0]
dyn_y_pred2 = results['mpc']['_tvp','dyn_obs_y_pred2',0]
dyn_x_pred2 = results['mpc']['_tvp','dyn_obs_x_pred2',0]


r = 0.5
sensor_r = 1

fig = plt.figure()
fig.set_dpi(100)
#fig.set_size_inches(7, 6.5)

ax = plt.axes(xlim=(-3, 3), ylim=(0, 13))
plt.axis('equal')
patch = plt.Circle((5, 8.5), 1, fc='y')
ax.plot(x,y)

patch_com= plt.Circle((x[0],y[0]), 0.01)
patch3= plt.Circle((-px[0]+x[0],-py[0]+y[0]), 0.01)

circle1 = Circle((5, 8.5), 1, alpha=0.5)
circle2 = Circle((8, 9), 0.7, alpha=0.5)
goal = Circle((0, 6), 0.1, color="green", label="goal")
#circle1 = Circle((0.5, 9.25), 0.5, alpha=0.5)
#circle2 = Circle((1.8, 9.25), 0.5, alpha=0.5)
#goal = Circle((1.5, 8.5), 0.1, color="red", alpha=0.5)
#ax.add_artist(circle1)
#ax.add_artist(circle2)
ax.add_artist(goal)

sensor = Circle((x[0],y[0]), sensor_r, edgecolor='gray', fc='None')

patchp = Ellipse(xy=(dyn_x_pred[0], dyn_y_pred[0]), width=r*2,
                        height=2*r, 
                        edgecolor='r', fc='None', lw=2, label="Prediction")

patchp2 = Ellipse(xy=(dyn_x_pred2[0], dyn_y_pred2[0]), width=r*2,
                        height=2*r, 
                        edgecolor='r', fc='None', lw=2, label="Prediction")

ax.add_patch(patchp)
#ax.add_patch(sensor)
ax.add_patch(patchp2)
#print(np.sqrt(((x[1]-8)/0.7*1.2)**2 + ((y[1]-dyn[0])/0.7*1.4)**2)-1)

patch = Circle((dyn_x[0], dyn_y[0]), r, fc='y', alpha=0.2, label="obstacle")
patchg = Circle((dyn_x[0], dyn_y[0]), gamma[0], edgecolor='blue', fc='None', alpha=0.2, label="obstacle")
patch_aug = Circle((dyn_x[0], dyn_y[0]), hn_aug[0], edgecolor='red', fc='None', alpha=0.2, label="obstacle")
patch2 = Circle((dyn_x2[0], dyn_y2[0]), r, fc='y', alpha=0.2, label="obstacle")

arrow = Arrow(x[0],y[0],0.15*cos(heading[0]),0.15*sin(heading[0]), width=0.2, fc="red")
ax.add_patch(arrow)

def init():
    ax.add_patch(patchp)
    return patchp, patch, patchp2, patch2, #sensor

def animate(i): 
    
    x_s = x[i]
    y_s = y[i]
    patch_com= plt.Circle((x_s,y_s), 0.08)
    sensor.set_center((x_s,y_s))
    x_f = px[i]+ x[i]
    y_f = py[i]+ y[i]
    patch3= plt.Circle((x_f,y_f), 0.08, fc='black')
    ax.add_artist(patch_com)
    ax.add_artist(patch3)

    arrow = plt.Arrow(x[i],y[i],0.15*cos(heading[i]),0.15*sin(heading[i]), width=0.2, fc="red")
    ax.add_patch(arrow)
    x_obs = dyn_x[i]
    y_obs = dyn_y[i]
    #patch = plt.Circle((x_obs, y_obs), 0.7, fc='y', alpha=0.9)
    patch.set_center((x_obs, y_obs))
     #just moving the plot and nnot over plotting
    x_obsp = dyn_x_pred[i]
    y_obsp = dyn_y_pred[i]
    patchp.set_center((x_obsp, y_obsp))
    patchp.height = 2*dyn_ry[i]*r
    patchp.width = 2*dyn_rx[i]*r
    patchg.set_center((x_obs, y_obs))
    patchg.radius=r+gamma[i]

    patch_aug.set_center((x_obsp, y_obsp))
    patch_aug.radius=r+hn_aug[i]

    x_obs2 = dyn_x2[i]
    y_obs2 = dyn_y2[i]
    #patch = plt.Circle((x_obs, y_obs), 0.7, fc='y', alpha=0.9)
    patch2.set_center((x_obs2, y_obs2)) #just moving the plot and nnot over plotting
    x_obsp2 = dyn_x_pred2[i]
    y_obsp2 = dyn_y_pred2[i]
    patchp2.set_center((x_obsp2, y_obsp2))
    patchp2.height = 2*dyn_ry2[i]*r
    patchp2.width = 2*dyn_rx2[i]*r

    
    #ax.add_artist(sensor)

    ax.add_artist(patch)
    #ax.add_artist(patchg)
    #ax.add_artist(patch_aug)
    ax.add_patch(patchp)

    ax.add_artist(patch2)

    ax.add_patch(patchp2)

    
    return patchp, patch, patchp2, patch2, patch_com, sensor
    
#plt.scatter(dyn_x,dyn_y, color='y')
    
#plt.legend()


ax.title.set_text('Uniform Random distrubtion with Prediction')
anim = FuncAnimation(fig, animate, init_func=init, interval=500)



plt.show()
