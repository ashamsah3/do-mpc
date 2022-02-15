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


def template_mpc(model):
    """
    --------------------------------------------------------------------------
    template_mpc: tuning parameters
    --------------------------------------------------------------------------
    """
    mpc = do_mpc.controller.MPC(model)

    setup_mpc = {
        'n_robust': 0,
        'n_horizon': 3,
        't_step': 1,
        'store_full_solution':True,
        #'nl_cons_check_colloc_points': True,
    }

    mpc.set_param(**setup_mpc)

    xg=0.5
    yg=9
    xdg=0
    ydg=0

    mterm = (-(model.x['x',1] - xdg)**2 + (model.x['x',3] - ydg)**2) + ((model.x['x',0] - xg)**2 + (model.x['x',2] - yg)**2)#model.aux['cost']
    lterm = (-(model.x['x',1] - xdg)**2 + (model.x['x',3] - ydg)**2) + ((model.x['x',0] - xg)**2 + (model.x['x',2] - yg)**2) #model.aux['cost'] # terminal cost
   # mterm = ((model.x['x',1] - 1)*(model.x['x',1] - 1)+ ((model.x['x',3] - 0)*(model.x['x',3] - 0)) + (model.x['x',2] - 0)*(model.x['x',2] - 0)) #model.aux['cost']
  #  lterm = ((model.x['x',1] - 1)*(model.x['x',1] - 1)+ ((model.x['x',3] - 0)*(model.x['x',3] - 0)) + (model.x['x',2] - 0)*(model.x['x',2] - 0)) #model.aux['cost'] # terminal cost

    mpc.set_objective(mterm=mterm, lterm=lterm)
    mpc.set_rterm(u=1e-4)

    max_x = np.array([[100.0], [1], [100.0], [1]])
    min_x = np.array([[-10.0], [-0.2], [-10.0], [-1]])
    max_u = np.array([[0.4], [0.4]])
    min_u = np.array([[0.0], [-0.4]])

    mpc.bounds['lower','_x','x'] =  min_x
    mpc.bounds['upper','_x','x'] =  max_x

    mpc.bounds['lower','_u','u'] =  min_u
    mpc.bounds['upper','_u','u'] =  max_u

    mpc.set_nl_cons('obstacles', -model.aux['obstacle_distance'], 0)
    #mpc.set_nl_cons('kin', -sqrt(model.u['u',0]**2 + model.u['u',1]**2), 0.1)
    #mpc.set_nl_cons('PSP2', -fabs(model.u['u',1]), ub=-0.05)



    mpc.setup()

    return mpc
