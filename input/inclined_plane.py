#! /usr/bin/env python
# Model parameters
# Test using the inclined plane analytical solution

#from numpy import *
#from netCDF4 import Dataset
#import time
import input_class
from input_class import *

# Declare model and init vailes
Model = Input(128,128,1)

#################################
#  Infomation of data in file   #
#################################
Model.description = 'Testing the inclined plane setup'
Model.history = 'Created ' + time.ctime(time.time())

###################################
#        Create dimensions        #
###################################
# Assigne value
Model.MaxIter = 1000 # Max number of multigrid iterations
Model.LevelIter[0] = 150

# General setup
Model.G=10; # Acceleration of Gravity, m/s^2
Model.XSize = 1000.0
Model.YSize = 1000.0
Model.VKoef=0.2;
Model.PKoef=0.2;
Model.KRelaxs=0.9; # 0.9
Model.KRelaxc=0.3; # Continuity equation 0.3
Model.update_model( )

################################
#    Setup viscosity array     #
################################
# Block settings
rhomedium=3000.0; # Medium density
etamedium=1e+20; # Medium viscosity, Pa*s

# Iteration/Multigrid parameters
alpha = 10*pi/180

# Grid points cycle
for i in range(0,int(Model.XNum[0])):
    for j in range(0,int(Model.YNum[0])):
        Model.Etas0[i,j] = etamedium
        Model.Etan0[i,j] = etamedium
        Model.P0[i,j] = 0

        if (i == 0 ):
            Model.Vx0[i,j] = 0
            Model.Vy0[i,j] = 0
            Model.P0[i,j] = 0
        if (i == Model.XNum[0]-1):
            Model.Vx0[i,j] = 0
            Model.Vy0[i,j] = 0
            Model.P0[i,j] = 0
        if (j == Model.YNum[0]-1):
            Model.Vx0[i,j] = 0
            Model.Vy0[i,j] = 0
            Model.P0[i,j] = 0
        if (j == 0):
            Model.Vx0[i,j] = 0
            Model.Vy0[i,j] = 0
            Model.P0[i,j] = 0
        
        Model.Rho0[i,j] = rhomedium

        if (i > 0 and i< Model.XNum[0]-1 and j > 0 and j < Model.YNum[0]-1):
            Model.Rx0[i,j] = -rhomedium*sin(alpha)*Model.G 
            Model.Ry0[i,j] = -rhomedium*cos(alpha)*Model.G


# Boundary conditions
# The indexing is like this
# [ Left | Right | Up | Down ]
# Boundaries are defined by
# 0: Free slip 1: No slip 2: Periodic
Model.BC = array( [2,2,0,1] )


Model.write_file("inclined_plane", 1)
