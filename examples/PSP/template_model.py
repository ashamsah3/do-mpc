#
#   This file is part of do-mpc
#
#   do-mpc: An environment for the easy, modular and efficient implementation of
#        robust nonlinear model predictive control
#
#   Copyright (c) 2014-2019 Sergio Lucia, Alexandru Tatulea-Codrean
#                        TU Dortmund. All rights reserved
#
#   do-mpc is free software: you can redistribute it and/or modify
#   it under the terms of the GNU Lesser General Public License as
#   published by the Free Software Foundation, either version 3
#   of the License, or (at your option) any later version.
#
#   do-mpc is distributed in the hope that it will be useful,
#   but WITHOUT ANY WARRANTY; without even the implied warranty of
#   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#   GNU Lesser General Public License for more details.
#
#   You should have received a copy of the GNU General Public License
#   along with do-mpc.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np
from casadi import *
from casadi.tools import *
import pdb
import sys
sys.path.append('../../')
import do_mpc


def template_model(obstacles, symvar_type='SX'):
    """
    --------------------------------------------------------------------------
    template_model: Variables / RHS / AUX
    --------------------------------------------------------------------------
    """
    model_type = 'discrete' # either 'discrete' or 'continuous'
    model = do_mpc.model.Model(model_type, symvar_type)

    # Simple oscillating masses example with two masses and two inputs.
    # States are the position and velocitiy of the two masses.

    # States struct (optimization variables):
    _x = model.set_variable(var_type='_x', var_name='x', shape=(4,1))

    # Input struct (optimization variables):
    _u = model.set_variable(var_type='_u', var_name='u', shape=(2,1))

    # Set expression. These can be used in the cost function, as non-linear constraints
    # or just to monitor another output.
    #model.set_expression(expr_name='cost', expr=sum1(_x**2))

    dyn_obs = model.set_variable('_tvp', 'dyn_obs')
    dyn_obs_x = model.set_variable('_tvp', 'dyn_obs_x')
    dyn_obs_y = model.set_variable('_tvp', 'dyn_obs_y')
    dyn_obs_x_pred = model.set_variable('_tvp', 'dyn_obs_x_pred')
    dyn_obs_y_pred = model.set_variable('_tvp', 'dyn_obs_y_pred')
    dyn_obs_rx = model.set_variable('_tvp', 'dyn_obs_rx')
    dyn_obs_ry = model.set_variable('_tvp', 'dyn_obs_ry')
    stance = model.set_variable('_tvp', 'stance')
    

    w=np.sqrt(9.81/0.8)
    T=0.4
    
    A = np.array([[1, np.sinh(w*T)/w, 0, 0],
                  [0, np.cosh(w*T), 0, 0],
                  [0, 0, 1, np.sinh(w*T)/w],
                  [0, 0, 0, np.cosh(w*T)]])


    B = np.array([[1-np.cosh(w*T), 0],
                  [-w*np.sinh(w*T), 0],
                  [0, 1-np.cosh(w*T)],
                  [0, -w*np.sinh(w*T)]])


    x_next = A@_x+B@_u
    model.set_rhs('x', x_next)

    # Calculations to avoid obstacles:

    # get x and y position 
    xk = model.x['x',0]
    yk = model.x['x',2]

    xk_n = x_next[0]
    yk_n = x_next[2]

    delta_xk = xk_n - xk
    delta_yk = yk_n - yk

    sin_th_k = delta_xk / sqrt(delta_xk*delta_xk + delta_yk*delta_yk)
    cos_th_k = delta_yk / sqrt(delta_xk*delta_xk + delta_yk*delta_yk)

    
    psp_safety = model.tvp['stance']*(fabs(model.x['x',3])-(fabs(model.u['u',1])*w)) #theorem2
    #kin_safety = model.tvp['stance']*(fabs(model.u['u',1])-0.15)#-0.01 everyting is less than pos 0.01
    kin_safety = (delta_xk**2 + delta_yk**2) - 0.5**2  
    kin_safety_min = 0.1**2 - (delta_xk**2 + delta_yk**2)
    grizzle1 = (sin_th_k* model.u['u',0] + cos_th_k* model.u['u',1]) - 0.3
    grizzle11 = -0.2 -(sin_th_k* model.u['u',0] + cos_th_k* model.u['u',1]) 
    grizzle2 = model.tvp['stance']*(-cos_th_k * model.u['u',0] + sin_th_k*model.u['u',1]) - (0.25)
    grizzle22= (0.05) - ((-cos_th_k*model.u['u',0] + sin_th_k*model.u['u',1])*model.tvp['stance'])
    #grizzle3 = (delta_xk**2 + delta_yk**2)-0.04

    obstacle_distance = []
    
    dyn_obstacle_distance = []

    h_xk11 = []
    h_xk1_n1 = []

    for obs in obstacles:
       #h_xk1= sqrt(((xk-obs['x']))**2+((yk-obs['y']))**2)-(obs['r']+0.2)#sqrt(((xk-obs['x']/obs['r']))**2+((yk-obs['y']/obs['r']))**2)-1 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       #h_xk2= sqrt(((xk-obs['x2']))**2+((yk-obs['y2']))**2)-(obs['r2']+0.2)#sqrt(((xk-obs['x2']/obs['r2']))**2+((yk-obs['y2']/obs['r2']))**2)-5 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       h_xk1= sqrt(((xk-obs['x'])/obs['r'])**2+((yk-obs['y'])/obs['r'])**2)-1#sqrt(((xk-obs['x']/obs['r']))**2+((yk-obs['y']/obs['r']))**2)-1 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       h_xk1_n= sqrt(((xk_n-obs['x'])/obs['r'])**2+((yk_n-obs['y'])/obs['r'])**2)-1
       #h_xk2= sqrt(((xk-obs['x2'])/obs['r2'])**2+((yk-obs['y2'])/obs['r2'])**2)-1#sqrt(((xk-obs['x2']/obs['r2']))**2+((yk-obs['y2']/obs['r2']))**2)-5 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       #h_xk3= sqrt(((xk_n-obs['x2'])/obs['r2']*1.2)**2 + ((yk_n-(model.tvp['dyn_obs']))/obs['r2']*1.4)**2)-1
       h_xk3= sqrt(((xk-model.tvp['dyn_obs_x'])/obs['r2'])**2 + ((yk-model.tvp['dyn_obs_y'])/obs['r2'])**2)-1
       #h_xk3_n= sqrt(((xk_n-model.tvp['dyn_obs_x'])/obs['r2'])**2 + ((yk_n-(model.tvp['dyn_obs_y']))/obs['r2'])**2)-1
       h_xk3_n= sqrt(((xk_n-model.tvp['dyn_obs_x_pred'])/(obs['r2']*model.tvp['dyn_obs_rx']))**2 + ((yk_n-(model.tvp['dyn_obs_y_pred']))/(obs['r2']*model.tvp['dyn_obs_ry']))**2)-1
       #h_xk3_n= sqrt(((xk_n-obs['x2'])/obs['r2']*1.2)**2 + ((yk_n-(model.tvp['dyn_obs']-0.3))/obs['r2']*1.6)**2)-1
       #h_xk= fmin(h_xk1,h_xk2)
       h_xk= ((1 - 0.1)*h_xk1 - h_xk1_n)
       full_hxk3 = ((1-0.7)*h_xk3 - h_xk3_n)
       h_xkmin = fmax(h_xk,full_hxk3)
       #obstacle_distance.extend([h_xk])
       obstacle_distance.extend([full_hxk3])
       #obstacle_distance.extend([h_xkmin])
      
      

    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))
    #model.set_expression('hk1',vertcat(*h_xk11))
    #model.set_expression('hk1_n',vertcat(*h_xk1_n1))
    model.set_expression('psp_safety',psp_safety)
    model.set_expression('kin_safety',kin_safety)
    model.set_expression('kin_safety_min',kin_safety_min)
    model.set_expression('grizzle1',grizzle1)
    model.set_expression('grizzle11',grizzle11)
    model.set_expression('grizzle2',grizzle2)
    model.set_expression('grizzle22',grizzle22)
    #model.set_expression('grizzle3',grizzle3)

    #h_xk3= sqrt(((xk-obs['x2']))**2+((yk-model.tvp['dyn_obs']))**2)-(obs['r2']+0.2)
    #h_xk3_n= sqrt(((xk_n-obs['x2']))**2+((yk_n-model.tvp['dyn_obs']))**2)-(obs['r2']+0.2)
    

    #dyn_obstacle_distance.extend([full_hxk3])
    #model.set_expression('dyn_obstacle_distance', vertcat(*dyn_obstacle_distance))

    

    model.setup()

    return model
