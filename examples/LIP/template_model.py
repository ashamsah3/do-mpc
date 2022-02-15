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

    w=np.sqrt(9.81/0.985)
    T=1

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
   


    obstacle_distance = []
    

    for obs in obstacles:
       h_xk1= sqrt(((xk-obs['x']))**2+((yk-obs['y']))**2)-(obs['r']+0.2)#sqrt(((xk-obs['x']/obs['r']))**2+((yk-obs['y']/obs['r']))**2)-1 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       h_xk2= sqrt(((xk-obs['x2']))**2+((yk-obs['y2']))**2)-(obs['r2']+0.2)#sqrt(((xk-obs['x2']/obs['r2']))**2+((yk-obs['y2']/obs['r2']))**2)-5 #((xk-obs['x'])**2+(yk-obs['y'])**2)-obs['r']*1.05 # np.sqrt(((xk-obs['y'])/obs['r'])**2 + ((yk-obs['y'])/obs['r'])**2)
       
       h_xk= fmin(h_xk1,h_xk2)
       obstacle_distance.extend([h_xk])
       #obstacle_distance.extend([h_xk1, h_xk2])
      


    model.set_expression('obstacle_distance',vertcat(*obstacle_distance))

    

    model.setup()

    return model
