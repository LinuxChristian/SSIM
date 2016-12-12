
#! /usr/bin/env python
# Model parameters
# Setup corresponds to "Hydrostatic pressure" test:

from numpy import *
#from Scientific.IO import NetCDF
from netCDF4 import Dataset
import time
import sys
import os
#import pytest

# External modules to import
#@pytest.mark.xfail
class Input:
   """
   This is a test of python docstrings

   """
   def __init__(self,XNum,YNum,Levels,debug):
      """
      Generates initial setup
      """
      #################################
      #  Infomation of data in file   #
      #################################
      self.description = 'Default input for SSIM'
      self.history = 'Created ' + time.ctime(time.time())

      ###################################
      #        Create dimensions        #
      ###################################
      # Assigne value
      self.MaxLevel = 8 # Max number of posible levels
      self.Levels = Levels # Number of multigrid levels
      if (self.Levels >= self.MaxLevel):
         print("The code only supports 8 levels!")
         exit()

      self.MaxIter = 0 # Max number of multigrid iterations
      self.Inum = 600 # Total number of multigrid iteration cycles
      self.startLevel = 0 # The first level of MG cycle (Only for debugging)
      self.IOUpdate = 1 # The interval where output is written. 1 is every multigrid cycle
      self.Equations = 0 # 0: 2D Full Stokes
                         # 1: 3D Full Stokes
                         # 2: iSOSIA
 
      # GPU settings
      # Thread and block size on level 0
      # NOTE:
      # Thread per block must be a even number to
      # get a Red-Black indexing between blocks!
      self.Threads = array( [16.0,16.0,0] ) # Remember to keep even
#      self.Threads = array( [8.0,8.0,0] ) # Remember to keep even

      self.Blocks = array( [ceil(XNum/self.Threads[0]), ceil(YNum/self.Threads[1]), 0] )

      #################################
      #    Test if input is valid     #
      #################################
#      if (mod(XNum,2) != 0 and ( self.Blocks[0] > 1 or self.Blocks[1] > 1 ) and mod(YNum,2) != 0 ):
#         print("Dimensions need to be a even number!")
#         exit()


      # General setup
      self.G=9.82; # Acceleration of Gravity, m/s^2
      self.PNorm=zeros( (self.Levels) ); # Pressure in the upermost, lwftmost (first) cell
      self.XSize=1000.0; # Model size, m
      self.YSize=1000.0;
      self.MeanDensity = 917.0 # Ice density (kg/m^3)

      # Relaxation coefficients for Gauss-Seidel iterations
      # Stokes equations
      self.KRelaxs=0.9; # 0.9
      self.KRelaxc=0.3; # Continuity equation 0.3
      self.VKoef=1.0;
      self.PKoef=1.0;

      # iSOSIA equations
      self.KRelaxv = 0.9;
      self.KRelaxs = 0.9;
      self.VKoef = 1.0;
      self.SKoef = 1.0;
      self.Gamma = 1e-15;
      self.AKoef = 1e-15;

      self.XNum = ones( (self.MaxLevel) )
      self.YNum = ones( (self.MaxLevel) )
      
      # Time between sync
      # Each block will run this many times
      # before writing to global memory
      self.SyncIter = ones( (self.Levels) )
         
      # Number of smoothing iterations for different levels
      self.LevelIter = zeros( (self.Levels) )
      self.LevelIter[0]=5
      for x in range(1,self.Levels):
         self.LevelIter[x]=self.LevelIter[0]*pow(2,x)
            
      # Defining resolutions for different levels
      if (debug):
         self.XNum[0] = XNum #vThreads[0]*vBlocks[0] # Add padding
         self.YNum[0] = YNum #vThreads[1]*vBlocks[1]
         for x in range(1,self.Levels):
            self.XNum[x] = self.XNum[0]
            self.YNum[x] = self.YNum[0]
      else:
         self.XNum[0] = XNum #vThreads[0]*vBlocks[0] # Add padding
         self.YNum[0] = YNum #vThreads[1]*vBlocks[1]
         for x in range(1,self.Levels):
            self.XNum[x] = floor((self.XNum[0]-1)/pow(2,x)+1)
            self.YNum[x] = floor((self.YNum[0]-1)/pow(2,x)+1)
            if ( (self.XNum[x]-1 < 2 or self.YNum[x]-1 < 2)):
               print(" ////////////////////////////////////////////////////////")
               print(" //                     FATAL ERROR                    //")
               print(" // To many levels or to low resolution on finest grid //")
               print(" ////////////////////////////////////////////////////////")
               exit()

                  
                  
      self.XStp = zeros( (self.Levels) )
      self.YStp = zeros( (self.Levels) )
                  
      # Defining gridsteps for all levels  Number 
      for x in range(0,self.Levels):
         # NOTE: CHANGES FROM -1 to -0
         self.XStp[x]=self.XSize/((self.XNum[x])); 
         self.YStp[x]=self.YSize/((self.YNum[x]));

      # Add padding
#      for x in range(0,self.Levels):
#         self.XNum[x] = self.XNum[x]+2
#         self.YNum[x] = self.YNum[x]+2         


      # Set restriction to use
      # DEFAULT: 2 - Bilinear
      self.RestrictOperator = 2

      # Set prolongation to use
      # DEFAULT: 0 - Bilinear
      self.ProlongOperator = 0

   def write_file ( self, FileName, verbose ):

      if (verbose != 0):
         print(" ////////////////////////////////////////////////////////")
         print(" //          CREATING INPUT FILE FOR SSIM              //")
         print(" ////////////////////////////////////////////////////////")

      # Test if output folder exists
      if FileName[0] == '/':
         InputFile = FileName+".nc"
      else:
         # If a relative path put it in output
         InputFile = "output/"+FileName+".nc"

         if not os.path.exists('output/'):
            os.makedirs('output/')

      PythonModule = "test/"+FileName+".py"

      if (verbose != 0):
         print("File: "+InputFile)
         print("Module to use: "+PythonModule)

      rootgrp = Dataset(InputFile, 'w', format='NETCDF4') # Open datafile for write


      ###################################
      #        Create variables         #
      ###################################
      # Dimensions
      rootgrp.createDimension('single',1)
      rootgrp.createDimension('dim3',3)
      rootgrp.createDimension('dim4',4)
      rootgrp.createDimension('dim8',8)
      rootgrp.createDimension('MultiLevels',self.MaxIter)
      rootgrp.createDimension('levels',self.Levels)
      rootgrp.createDimension('dLevels',None)

      # Level dimensions
      rootgrp.createDimension('dXNum0',self.XNum[0])
      rootgrp.createDimension('dYNum0',self.YNum[0])
      rootgrp.createDimension('dXNum1',self.XNum[1])
      rootgrp.createDimension('dYNum1',self.YNum[1])
      rootgrp.createDimension('dXNum2',self.XNum[2])
      rootgrp.createDimension('dYNum2',self.YNum[2])
      rootgrp.createDimension('dXNum3',self.XNum[3])
      rootgrp.createDimension('dYNum3',self.YNum[3])
      rootgrp.createDimension('dXNum4',self.XNum[4])
      rootgrp.createDimension('dYNum4',self.YNum[4])
      rootgrp.createDimension('dXNum5',self.XNum[5])
      rootgrp.createDimension('dYNum5',self.YNum[5])
      rootgrp.createDimension('dXNum6',self.XNum[6])
      rootgrp.createDimension('dYNum6',self.YNum[6])
      rootgrp.createDimension('dXNum7',self.XNum[7])
      rootgrp.createDimension('dYNum7',self.YNum[7])

      # Variables
      levels    = rootgrp.createVariable('levels'   ,dtype('int32')  ,('single'))
      startLevel = rootgrp.createVariable('startLevel'   ,dtype('int32')  ,('single'))
      maxIter   = rootgrp.createVariable('maxIter'  ,dtype('int32')  ,('single'))
      xSize     = rootgrp.createVariable('xSize'    ,dtype('double') ,('single'))
      ySize     = rootgrp.createVariable('ySize'    ,dtype('double') ,('single'))
      threads   = rootgrp.createVariable('threads'  ,dtype('int')    ,('dim3'))
      blocks    = rootgrp.createVariable('blocks'   ,dtype('int')    ,('dim3'))
      xNum      = rootgrp.createVariable('xNum'     ,dtype('int32')  ,('dim8'))
      yNum      = rootgrp.createVariable('yNum'     ,dtype('int32')  ,('dim8'))
      levelIter = rootgrp.createVariable('levelIter',dtype('int')    ,('levels'))
      xStp      = rootgrp.createVariable('xStp'     ,dtype('double') ,('levels'))
      yStp      = rootgrp.createVariable('yStp'     ,dtype('double') ,('levels'))
      BC        = rootgrp.createVariable('BC'       ,dtype('int')    ,('dim4'))
      Equations = rootgrp.createVariable('Equations',dtype('int')    ,('single'))
      G         = rootgrp.createVariable('G'        ,dtype('double') ,('single'))
      PNorm     = rootgrp.createVariable('PNorm'    ,dtype('double') ,('levels'))
      syncIter  = rootgrp.createVariable('syncIter' ,dtype('int')    ,('levels'))
      ioUpdate  = rootgrp.createVariable('IOUpdate' ,dtype('int')    ,('single'))
      restrictOperator = rootgrp.createVariable('RestrictOperator' ,dtype('int')    ,('single'))
      prolongOperator = rootgrp.createVariable('ProlongOperator' ,dtype('int')    ,('single'))

      ## Physical dimensions ##
      if (self.Equations == 0):
         pKoef     = rootgrp.createVariable('pKoef'    ,dtype('double') ,('single'))
         vKoef     = rootgrp.createVariable('vKoef'    ,dtype('double') ,('single'))
         relaxs    = rootgrp.createVariable('relaxs'   ,dtype('float')  ,('single'))
         relaxc    = rootgrp.createVariable('relaxc'   ,dtype('float')  ,('single'))
         
         # Level 0
         vx0   = rootgrp.createVariable('vx0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         vy0   = rootgrp.createVariable('vy0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         p0    = rootgrp.createVariable('p0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         rx0   = rootgrp.createVariable('rx0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         ry0   = rootgrp.createVariable('ry0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         rc0   = rootgrp.createVariable('rc0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         resx0 = rootgrp.createVariable('resx0' ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         resy0 = rootgrp.createVariable('resy0' ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         resc0 = rootgrp.createVariable('resc0' ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         etan0 = rootgrp.createVariable('etan0' ,dtype('double') ,('dXNum0','dYNum0'))
         etas0 = rootgrp.createVariable('etas0' ,dtype('double') ,('dXNum0','dYNum0'))      
         rho0  = rootgrp.createVariable('rho0'  ,dtype('double') ,('dXNum0','dYNum0'))
         
         # Level 1
         vx1   = rootgrp.createVariable('vx1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         vy1   = rootgrp.createVariable('vy1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         p1    = rootgrp.createVariable('p1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         rx1   = rootgrp.createVariable('rx1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         ry1   = rootgrp.createVariable('ry1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         rc1   = rootgrp.createVariable('rc1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         resx1 = rootgrp.createVariable('resx1' ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         resy1 = rootgrp.createVariable('resy1' ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         resc1 = rootgrp.createVariable('resc1' ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         etan1 = rootgrp.createVariable('etan1' ,dtype('double') ,('dXNum1','dYNum1'))
         etas1 = rootgrp.createVariable('etas1' ,dtype('double') ,('dXNum1','dYNum1'))      
         
         # Level 2
         vx2   = rootgrp.createVariable('vx2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         vy2   = rootgrp.createVariable('vy2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         p2    = rootgrp.createVariable('p2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         rx2   = rootgrp.createVariable('rx2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         ry2   = rootgrp.createVariable('ry2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         rc2   = rootgrp.createVariable('rc2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         resx2 = rootgrp.createVariable('resx2' ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         resy2 = rootgrp.createVariable('resy2' ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         resc2 = rootgrp.createVariable('resc2' ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         etan2 = rootgrp.createVariable('etan2' ,dtype('double') ,('dXNum2','dYNum2'))
         etas2 = rootgrp.createVariable('etas2' ,dtype('double') ,('dXNum2','dYNum2'))      
         
         # Level 3
         vx3   = rootgrp.createVariable('vx3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         vy3   = rootgrp.createVariable('vy3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         p3    = rootgrp.createVariable('p3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         rx3   = rootgrp.createVariable('rx3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         ry3   = rootgrp.createVariable('ry3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         rc3   = rootgrp.createVariable('rc3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         resx3 = rootgrp.createVariable('resx3' ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         resy3 = rootgrp.createVariable('resy3' ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         resc3 = rootgrp.createVariable('resc3' ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         etan3 = rootgrp.createVariable('etan3' ,dtype('double') ,('dXNum3','dYNum3'))
         etas3 = rootgrp.createVariable('etas3' ,dtype('double') ,('dXNum3','dYNum3'))      
         
         # Level 4
         vx4   = rootgrp.createVariable('vx4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         vy4   = rootgrp.createVariable('vy4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         p4    = rootgrp.createVariable('p4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         rx4   = rootgrp.createVariable('rx4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         ry4   = rootgrp.createVariable('ry4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         rc4   = rootgrp.createVariable('rc4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         resx4 = rootgrp.createVariable('resx4' ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         resy4 = rootgrp.createVariable('resy4' ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         resc4 = rootgrp.createVariable('resc4' ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         etan4 = rootgrp.createVariable('etan4' ,dtype('double') ,('dXNum4','dYNum4'))
         etas4 = rootgrp.createVariable('etas4' ,dtype('double') ,('dXNum4','dYNum4'))      
         
         # Level 5
         vx5   = rootgrp.createVariable('vx5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         vy5   = rootgrp.createVariable('vy5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         p5    = rootgrp.createVariable('p5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         rx5   = rootgrp.createVariable('rx5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         ry5   = rootgrp.createVariable('ry5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         rc5   = rootgrp.createVariable('rc5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         resx5 = rootgrp.createVariable('resx5' ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         resy5 = rootgrp.createVariable('resy5' ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         resc5 = rootgrp.createVariable('resc5' ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         etan5 = rootgrp.createVariable('etan5' ,dtype('double') ,('dXNum5','dYNum5'))
         etas5 = rootgrp.createVariable('etas5' ,dtype('double') ,('dXNum5','dYNum5'))      
         
         # Level 6
         vx6   = rootgrp.createVariable('vx6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         vy6   = rootgrp.createVariable('vy6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         p6    = rootgrp.createVariable('p6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         rx6   = rootgrp.createVariable('rx6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         ry6   = rootgrp.createVariable('ry6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         rc6   = rootgrp.createVariable('rc6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         resx6 = rootgrp.createVariable('resx6' ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         resy6 = rootgrp.createVariable('resy6' ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         resc6 = rootgrp.createVariable('resc6' ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         etan6 = rootgrp.createVariable('etan6' ,dtype('double') ,('dXNum6','dYNum6'))
         etas6 = rootgrp.createVariable('etas6' ,dtype('double') ,('dXNum6','dYNum6'))      
         
         # Level 7
         vx7   = rootgrp.createVariable('vx7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         vy7   = rootgrp.createVariable('vy7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         p7    = rootgrp.createVariable('p7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         rx7   = rootgrp.createVariable('rx7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         ry7   = rootgrp.createVariable('ry7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         rc7   = rootgrp.createVariable('rc7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         resx7 = rootgrp.createVariable('resx7' ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         resy7 = rootgrp.createVariable('resy7' ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         resc7 = rootgrp.createVariable('resc7' ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         etan7 = rootgrp.createVariable('etan7' ,dtype('double') ,('dXNum7','dYNum7'))
         etas7 = rootgrp.createVariable('etas7' ,dtype('double') ,('dXNum7','dYNum7'))      


      if (self.Equations == 2):
         VKoef   = rootgrp.createVariable('vKoef'   ,dtype('double') ,('single'))
         SKoef   = rootgrp.createVariable('sKoef'   ,dtype('double') ,('single'))
         KRelaxv = rootgrp.createVariable('kRelaxv' ,dtype('double') ,('single'))
         KRelaxs = rootgrp.createVariable('kRelaxs' ,dtype('double') ,('single'))
         gamma   = rootgrp.createVariable('gamma'   ,dtype('double') ,('single'))
         aKoef   = rootgrp.createVariable('aKoef'   ,dtype('double') ,('single'))

         # Level 0
         IceTopo0 = rootgrp.createVariable('iceTopo0' ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         BedTopo0 = rootgrp.createVariable('bedTopo0' ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         vx0      = rootgrp.createVariable('vx0'      ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         vy0      = rootgrp.createVariable('vy0'      ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         sxx0     = rootgrp.createVariable('sxx0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         syy0     = rootgrp.createVariable('syy0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         sxy0     = rootgrp.createVariable('sxy0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         exx0     = rootgrp.createVariable('exx0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         eyy0     = rootgrp.createVariable('eyy0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         exy0     = rootgrp.createVariable('exy0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         exym0    = rootgrp.createVariable('exym0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         ezz0     = rootgrp.createVariable('ezz0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))         
         taue0    = rootgrp.createVariable('taue0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         taueR0   = rootgrp.createVariable('taueR0'   ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         vxR0     = rootgrp.createVariable('vxR0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         vyR0     = rootgrp.createVariable('vyR0'     ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         resx0    = rootgrp.createVariable('resx0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         resy0    = rootgrp.createVariable('resy0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))
         rest0    = rootgrp.createVariable('rest0'    ,dtype('double') ,('dLevels','dXNum0','dYNum0'))


         # Level 1
         IceTopo1 = rootgrp.createVariable('iceTopo1' ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         BedTopo1 = rootgrp.createVariable('bedTopo1' ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         vx1      = rootgrp.createVariable('vx1'      ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         vy1      = rootgrp.createVariable('vy1'      ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         sxx1     = rootgrp.createVariable('sxx1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         syy1     = rootgrp.createVariable('syy1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         sxy1     = rootgrp.createVariable('sxy1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         exx1     = rootgrp.createVariable('exx1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         eyy1     = rootgrp.createVariable('eyy1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         exy1     = rootgrp.createVariable('exy1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         exym1    = rootgrp.createVariable('exym1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         ezz1     = rootgrp.createVariable('ezz1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))         
         taue1    = rootgrp.createVariable('taue1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         taueR1   = rootgrp.createVariable('taueR1'   ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         vxR1     = rootgrp.createVariable('vxR1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         vyR1     = rootgrp.createVariable('vyR1'     ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         resx1    = rootgrp.createVariable('resx1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         resy1    = rootgrp.createVariable('resy1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))
         rest1    = rootgrp.createVariable('rest1'    ,dtype('double') ,('dLevels','dXNum1','dYNum1'))

         # Level 2
         IceTopo2 = rootgrp.createVariable('iceTopo2' ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         BedTopo2 = rootgrp.createVariable('bedTopo2' ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         vx2      = rootgrp.createVariable('vx2'      ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         vy2      = rootgrp.createVariable('vy2'      ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         sxx2     = rootgrp.createVariable('sxx2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         syy2     = rootgrp.createVariable('syy2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         sxy2     = rootgrp.createVariable('sxy2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         exx2     = rootgrp.createVariable('exx2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         eyy2     = rootgrp.createVariable('eyy2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         exy2     = rootgrp.createVariable('exy2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         exym2    = rootgrp.createVariable('exym2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         ezz2     = rootgrp.createVariable('ezz2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))         
         taue2    = rootgrp.createVariable('taue2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         taueR2   = rootgrp.createVariable('taueR2'   ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         vxR2     = rootgrp.createVariable('vxR2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         vyR2     = rootgrp.createVariable('vyR2'     ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         resx2    = rootgrp.createVariable('resx2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         resy2    = rootgrp.createVariable('resy2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))
         rest2    = rootgrp.createVariable('rest2'    ,dtype('double') ,('dLevels','dXNum2','dYNum2'))

         # Level 3
         IceTopo3 = rootgrp.createVariable('iceTopo3' ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         BedTopo3 = rootgrp.createVariable('bedTopo3' ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         vx3      = rootgrp.createVariable('vx3'      ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         vy3      = rootgrp.createVariable('vy3'      ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         sxx3     = rootgrp.createVariable('sxx3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         syy3     = rootgrp.createVariable('syy3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         sxy3     = rootgrp.createVariable('sxy3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         exx3     = rootgrp.createVariable('exx3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         eyy3     = rootgrp.createVariable('eyy3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         exy3     = rootgrp.createVariable('exy3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         exym3    = rootgrp.createVariable('exym3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         ezz3     = rootgrp.createVariable('ezz3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))         
         taue3    = rootgrp.createVariable('taue3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         taueR3   = rootgrp.createVariable('taueR3'   ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         vxR3     = rootgrp.createVariable('vxR3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         vyR3     = rootgrp.createVariable('vyR3'     ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         resx3    = rootgrp.createVariable('resx3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         resy3    = rootgrp.createVariable('resy3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))
         rest3    = rootgrp.createVariable('rest3'    ,dtype('double') ,('dLevels','dXNum3','dYNum3'))

         # Level 4
         IceTopo4 = rootgrp.createVariable('iceTopo4' ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         BedTopo4 = rootgrp.createVariable('bedTopo4' ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         vx4      = rootgrp.createVariable('vx4'      ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         vy4      = rootgrp.createVariable('vy4'      ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         sxx4     = rootgrp.createVariable('sxx4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         syy4     = rootgrp.createVariable('syy4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         sxy4     = rootgrp.createVariable('sxy4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         exx4     = rootgrp.createVariable('exx4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         eyy4     = rootgrp.createVariable('eyy4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         exy4     = rootgrp.createVariable('exy4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         exym4    = rootgrp.createVariable('exym4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         ezz4     = rootgrp.createVariable('ezz4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))         
         taue4    = rootgrp.createVariable('taue4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         taueR4   = rootgrp.createVariable('taueR4'   ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         vxR4     = rootgrp.createVariable('vxR4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         vyR4     = rootgrp.createVariable('vyR4'     ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         resx4    = rootgrp.createVariable('resx4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         resy4    = rootgrp.createVariable('resy4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))
         rest4    = rootgrp.createVariable('rest4'    ,dtype('double') ,('dLevels','dXNum4','dYNum4'))

         # Level 5
         IceTopo5 = rootgrp.createVariable('iceTopo5' ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         BedTopo5 = rootgrp.createVariable('bedTopo5' ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         vx5      = rootgrp.createVariable('vx5'      ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         vy5      = rootgrp.createVariable('vy5'      ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         sxx5     = rootgrp.createVariable('sxx5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         syy5     = rootgrp.createVariable('syy5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         sxy5     = rootgrp.createVariable('sxy5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         exx5     = rootgrp.createVariable('exx5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         eyy5     = rootgrp.createVariable('eyy5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         exy5     = rootgrp.createVariable('exy5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         exym5    = rootgrp.createVariable('exym5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         ezz5     = rootgrp.createVariable('ezz5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))         
         taue5    = rootgrp.createVariable('taue5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         taueR5   = rootgrp.createVariable('taueR5'   ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         vxR5     = rootgrp.createVariable('vxR5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         vyR5     = rootgrp.createVariable('vyR5'     ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         resx5    = rootgrp.createVariable('resx5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         resy5    = rootgrp.createVariable('resy5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))
         rest5    = rootgrp.createVariable('rest5'    ,dtype('double') ,('dLevels','dXNum5','dYNum5'))

         # Level 6
         IceTopo6 = rootgrp.createVariable('iceTopo6' ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         BedTopo6 = rootgrp.createVariable('bedTopo6' ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         vx6      = rootgrp.createVariable('vx6'      ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         vy6      = rootgrp.createVariable('vy6'      ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         sxx6     = rootgrp.createVariable('sxx6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         syy6     = rootgrp.createVariable('syy6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         sxy6     = rootgrp.createVariable('sxy6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         exx6     = rootgrp.createVariable('exx6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         eyy6     = rootgrp.createVariable('eyy6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         exy6     = rootgrp.createVariable('exy6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         exym6    = rootgrp.createVariable('exym6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         ezz6     = rootgrp.createVariable('ezz6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))         
         taue6    = rootgrp.createVariable('taue6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         taueR6   = rootgrp.createVariable('taueR6'   ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         vxR6     = rootgrp.createVariable('vxR6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         vyR6     = rootgrp.createVariable('vyR6'     ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         resx6    = rootgrp.createVariable('resx6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         resy6    = rootgrp.createVariable('resy6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))
         rest6    = rootgrp.createVariable('rest6'    ,dtype('double') ,('dLevels','dXNum6','dYNum6'))


         # Level 7
         IceTopo7 = rootgrp.createVariable('iceTopo7' ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         BedTopo7 = rootgrp.createVariable('bedTopo7' ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         vx7      = rootgrp.createVariable('vx7'      ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         vy7      = rootgrp.createVariable('vy7'      ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         sxx7     = rootgrp.createVariable('sxx7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         syy7     = rootgrp.createVariable('syy7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         sxy7     = rootgrp.createVariable('sxy7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         exx7     = rootgrp.createVariable('exx7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         eyy7     = rootgrp.createVariable('eyy7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         exy7     = rootgrp.createVariable('exy7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         exym7    = rootgrp.createVariable('exym7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         ezz7     = rootgrp.createVariable('ezz7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))         
         taue7    = rootgrp.createVariable('taue7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         taueR7   = rootgrp.createVariable('taueR7'   ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         vxR7     = rootgrp.createVariable('vxR7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         vyR7     = rootgrp.createVariable('vyR7'     ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         resx7    = rootgrp.createVariable('resx7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         resy7    = rootgrp.createVariable('resy7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))
         rest7    = rootgrp.createVariable('rest7'    ,dtype('double') ,('dLevels','dXNum7','dYNum7'))


      # Write general variables
      levels[:]    = self.Levels
      startLevel[:]= self.startLevel
      maxIter[:]   = self.MaxIter
      xSize[:]     = self.XSize
      ySize[:]     = self.YSize
      threads[:]   = self.Threads
      blocks[:]    = self.Blocks
      xNum[:]      = self.XNum
      yNum[:]      = self.YNum
      levelIter[:] = self.LevelIter
      xStp[:]      = self.XStp
      yStp[:]      = self.YStp
      BC[:]        = self.BC
      Equations[:] = self.Equations
      G[:]         = self.G
      syncIter[:]  = self.SyncIter
      ioUpdate[:]  = self.IOUpdate
      restrictOperator[:] = self.RestrictOperator
      prolongOperator[:] = self.ProlongOperator


      ## Physical Variables for iSOSIA equations ##
      if (self.Equations == 2):
         VKoef[:]      = self.VKoef
         SKoef[:]      = self.SKoef
         KRelaxv[:]    = self.KRelaxv
         KRelaxs[:]    = self.KRelaxs
         gamma[:]      = self.Gamma
         aKoef[:]      = self.AKoef

         # Level 0
         IceTopo0[0,:,:] = self.IceTopo0
         BedTopo0[0,:,:] = self.BedTopo0
         vx0[0,:,:]      = self.Vx0
         vy0[0,:,:]      = self.Vy0
         sxx0[0,:,:]     = self.Sxx0
         syy0[0,:,:]     = self.Syy0
         sxy0[0,:,:]     = self.Sxy0
         exx0[0,:,:]     = self.Exx0
         eyy0[0,:,:]     = self.Eyy0
         exy0[0,:,:]     = self.Exy0
         exym0[0,:,:]    = self.Exym0
         ezz0[0,:,:]     = self.Ezz0
         taue0[0,:,:]    = self.Taue0
         taueR0[0,:,:]   = self.TaueR0
         vxR0[0,:,:]     = self.VxR0
         vyR0[0,:,:]     = self.VyR0
         resx0[0,:,:]    = self.Resx0
         resy0[0,:,:]    = self.Resy0
         rest0[0,:,:]    = self.Rest0

         # Level 1
         IceTopo1[0,:,:] = self.IceTopo1
         BedTopo1[0,:,:] = self.BedTopo1
         vx1[0,:,:]      = self.Vx1
         vy1[0,:,:]      = self.Vy1
         sxx1[0,:,:]     = self.Sxx1
         syy1[0,:,:]     = self.Syy1
         sxy1[0,:,:]     = self.Sxy1
         exx1[0,:,:]     = self.Exx1
         eyy1[0,:,:]     = self.Eyy1
         exy1[0,:,:]     = self.Exy1
         exym1[0,:,:]    = self.Exym1
         ezz1[0,:,:]     = self.Ezz1
         taue1[0,:,:]    = self.Taue1
         taueR1[0,:,:]   = self.TaueR1
         vxR1[0,:,:]     = self.VxR1
         vyR1[0,:,:]     = self.VyR1
         resx1[0,:,:]    = self.Resx1
         resy1[0,:,:]    = self.Resy1
         rest1[0,:,:]    = self.Rest1


         # Level 2
         IceTopo2[0,:,:] = self.IceTopo2
         BedTopo2[0,:,:] = self.BedTopo2
         vx2[0,:,:]      = self.Vx2
         vy2[0,:,:]      = self.Vy2
         sxx2[0,:,:]     = self.Sxx2
         syy2[0,:,:]     = self.Syy2
         sxy2[0,:,:]     = self.Sxy2
         exx2[0,:,:]     = self.Exx2
         eyy2[0,:,:]     = self.Eyy2
         exy2[0,:,:]     = self.Exy2
         exym2[0,:,:]    = self.Exym2
         ezz2[0,:,:]     = self.Ezz2
         taue2[0,:,:]    = self.Taue2
         taueR2[0,:,:]   = self.TaueR2
         vxR2[0,:,:]     = self.VxR2
         vyR2[0,:,:]     = self.VyR2
         resx2[0,:,:]    = self.Resx2
         resy2[0,:,:]    = self.Resy2
         rest2[0,:,:]    = self.Rest2

         # Level 3
         IceTopo3[0,:,:] = self.IceTopo3
         BedTopo3[0,:,:] = self.BedTopo3
         vx3[0,:,:]      = self.Vx3
         vy3[0,:,:]      = self.Vy3
         sxx3[0,:,:]     = self.Sxx3
         syy3[0,:,:]     = self.Syy3
         sxy3[0,:,:]     = self.Sxy3
         exx3[0,:,:]     = self.Exx3
         eyy3[0,:,:]     = self.Eyy3
         exy3[0,:,:]     = self.Exy3
         exym3[0,:,:]    = self.Exym3
         ezz3[0,:,:]     = self.Ezz3
         taue3[0,:,:]    = self.Taue3
         taueR3[0,:,:]   = self.TaueR3
         vxR3[0,:,:]     = self.VxR3
         vyR3[0,:,:]     = self.VyR3
         resx3[0,:,:]    = self.Resx3
         resy3[0,:,:]    = self.Resy3
         rest3[0,:,:]    = self.Rest3

         # Level 4
         IceTopo4[0,:,:] = self.IceTopo4
         BedTopo4[0,:,:] = self.BedTopo4
         vx4[0,:,:]      = self.Vx4
         vy4[0,:,:]      = self.Vy4
         sxx4[0,:,:]     = self.Sxx4
         syy4[0,:,:]     = self.Syy4
         sxy4[0,:,:]     = self.Sxy4
         exx4[0,:,:]     = self.Exx4
         eyy4[0,:,:]     = self.Eyy4
         exy4[0,:,:]     = self.Exy4
         exym4[0,:,:]    = self.Exym4
         ezz4[0,:,:]     = self.Ezz4
         taue4[0,:,:]    = self.Taue4
         taueR4[0,:,:]   = self.TaueR4
         vxR4[0,:,:]     = self.VxR4
         vyR4[0,:,:]     = self.VyR4
         resx4[0,:,:]    = self.Resx4
         resy4[0,:,:]    = self.Resy4
         rest4[0,:,:]    = self.Rest4

         # Level 5
         IceTopo5[0,:,:] = self.IceTopo5
         BedTopo5[0,:,:] = self.BedTopo5
         vx5[0,:,:]      = self.Vx5
         vy5[0,:,:]      = self.Vy5
         sxx5[0,:,:]     = self.Sxx5
         syy5[0,:,:]     = self.Syy5
         sxy5[0,:,:]     = self.Sxy5
         exx5[0,:,:]     = self.Exx5
         eyy5[0,:,:]     = self.Eyy5
         exy5[0,:,:]     = self.Exy5
         exym5[0,:,:]    = self.Exym5
         ezz5[0,:,:]     = self.Ezz5
         taue5[0,:,:]    = self.Taue5
         taueR5[0,:,:]   = self.TaueR5
         vxR5[0,:,:]     = self.VxR5
         vyR5[0,:,:]     = self.VyR5
         resx5[0,:,:]    = self.Resx5
         resy5[0,:,:]    = self.Resy5
         rest5[0,:,:]    = self.Rest5

         # Level 6
         IceTopo6[0,:,:] = self.IceTopo6
         BedTopo6[0,:,:] = self.BedTopo6
         vx6[0,:,:]      = self.Vx6
         vy6[0,:,:]      = self.Vy6
         sxx6[0,:,:]     = self.Sxx6
         syy6[0,:,:]     = self.Syy6
         sxy6[0,:,:]     = self.Sxy6
         exx6[0,:,:]     = self.Exx6
         eyy6[0,:,:]     = self.Eyy6
         exy6[0,:,:]     = self.Exy6
         exym6[0,:,:]    = self.Exym6
         ezz6[0,:,:]     = self.Ezz6
         taue6[0,:,:]    = self.Taue6
         taueR6[0,:,:]   = self.TaueR6
         vxR6[0,:,:]     = self.VxR6
         vyR6[0,:,:]     = self.VyR6
         resx6[0,:,:]    = self.Resx6
         resy6[0,:,:]    = self.Resy6
         rest6[0,:,:]    = self.Rest6

         # Level 7
         IceTopo7[0,:,:] = self.IceTopo7
         BedTopo7[0,:,:] = self.BedTopo7
         vx7[0,:,:]      = self.Vx7
         vy7[0,:,:]      = self.Vy7
         sxx7[0,:,:]     = self.Sxx7
         syy7[0,:,:]     = self.Syy7
         sxy7[0,:,:]     = self.Sxy7
         exx7[0,:,:]     = self.Exx7
         eyy7[0,:,:]     = self.Eyy7
         exy7[0,:,:]     = self.Exy7
         exym7[0,:,:]    = self.Exym7
         ezz7[0,:,:]     = self.Ezz7
         taue7[0,:,:]    = self.Taue7
         taueR7[0,:,:]   = self.TaueR7
         vxR7[0,:,:]     = self.VxR7
         vyR7[0,:,:]     = self.VyR7
         resx7[0,:,:]    = self.Resx7
         resy7[0,:,:]    = self.Resy7
         rest7[0,:,:]    = self.Rest7


      ## Write physical variables ##
      if (self.Equations == 0):
         PNorm[:]     = self.PNorm
         pKoef[:]     = self.PKoef
         vKoef[:]     = self.VKoef
         relaxs[:]    = self.KRelaxs
         relaxc[:]    = self.KRelaxc

         # Level 0
         vx0[0,:,:]   = self.Vx0
         vy0[0,:,:]   = self.Vy0
         p0[0,:,:]    = self.P0
         rx0[0,:,:]   = self.Rx0
         ry0[0,:,:]   = self.Ry0
         rc0[0,:,:]   = self.Rc0
         resx0[0,:,:] = self.Resx0
         resy0[0,:,:] = self.Resy0
         resc0[0,:,:] = self.Resc0
         etan0[:]     = self.Etan0
         etas0[:]     = self.Etas0
         rho0[:]       = self.Rho0
         
         # Level 1
         vx1[0,:,:]   = self.Vx1
         vy1[0,:,:]   = self.Vy1
         p1[0,:,:]    = self.P1
         rx1[0,:,:]   = self.Rx1
         ry1[0,:,:]   = self.Ry1
         rc1[0,:,:]   = self.Rc1
         resx1[0,:,:] = self.Resx1
         resy1[0,:,:] = self.Resy1
         resc1[0,:,:] = self.Resc1
         etan1[:]     = self.Etan1
         etas1[:]     = self.Etas1
         
         # Level 2
         vx2[0,:,:]   = self.Vx2
         vy2[0,:,:]   = self.Vy2
         p2[0,:,:]    = self.P2
         rx2[0,:,:]   = self.Rx2
         ry2[0,:,:]   = self.Ry2
         rc2[0,:,:]   = self.Rc2
         resx2[0,:,:] = self.Resx2
         resy2[0,:,:] = self.Resy2
         resc2[0,:,:] = self.Resc2
         etan2[:]     = self.Etan2
         etas2[:]     = self.Etas2
         
         # Level 3
         vx3[0,:,:]   = self.Vx3
         vy3[0,:,:]   = self.Vy3
         p3[0,:,:]    = self.P3
         rx3[0,:,:]   = self.Rx3
         ry3[0,:,:]   = self.Ry3
         rc3[0,:,:]   = self.Rc3
         resx3[0,:,:] = self.Resx3
         resy3[0,:,:] = self.Resy3
         resc3[0,:,:] = self.Resc3
         etan3[:]     = self.Etan3
         etas3[:]     = self.Etas3
         
         # Level 4
         vx4[0,:,:]   = self.Vx4
         vy4[0,:,:]   = self.Vy4
         p4[0,:,:]    = self.P4
         rx4[0,:,:]   = self.Rx4
         ry4[0,:,:]   = self.Ry4
         rc4[0,:,:]   = self.Rc4
         resx4[0,:,:] = self.Resx4
         resy4[0,:,:] = self.Resy4
         resc4[0,:,:] = self.Resc4
         etan4[:]     = self.Etan4
         etas4[:]     = self.Etas4
         
         # Level 5
         vx5[0,:,:]   = self.Vx5
         vy5[0,:,:]   = self.Vy5
         p5[0,:,:]    = self.P5
         rx5[0,:,:]   = self.Rx5
         ry5[0,:,:]   = self.Ry5
         rc5[0,:,:]   = self.Rc5
         resx5[0,:,:] = self.Resx5
         resy5[0,:,:] = self.Resy5
         resc5[0,:,:] = self.Resc5
         etan5[:]     = self.Etan5
         etas5[:]     = self.Etas5
         
         # Level 6
         vx6[0,:,:]   = self.Vx6
         vy6[0,:,:]   = self.Vy6
         p6[0,:,:]    = self.P6
         rx6[0,:,:]   = self.Rx6
         ry6[0,:,:]   = self.Ry6
         rc6[0,:,:]   = self.Rc6
         resx6[0,:,:] = self.Resx6
         resy6[0,:,:] = self.Resy6
         resc6[0,:,:] = self.Resc6
         etan6[:]     = self.Etan6
         etas6[:]     = self.Etas6
         
         # Level 7
         vx7[0,:,:]   = self.Vx7
         vy7[0,:,:]   = self.Vy7
         p7[0,:,:]    = self.P7
         rx7[0,:,:]   = self.Rx7
         ry7[0,:,:]   = self.Ry7
         rc7[0,:,:]   = self.Rc7
         resx7[0,:,:] = self.Resx7
         resy7[0,:,:] = self.Resy7
         resc7[0,:,:] = self.Resc7
         etan7[:]     = self.Etan7
         etas7[:]     = self.Etas7
         
      rootgrp.close() # Close datafile

      if (verbose != 0):
         print(" ////////////////////////////////////////////////////////")
         print(" //        OUTPUT FILE WRITTEN WITH SUCESS             //")
         print(" ////////////////////////////////////////////////////////")


   def update_model (self):
      # Update model with users setting

      self.Blocks = array( [ceil(self.XNum[0]/self.Threads[0]), ceil(self.YNum[0]/self.Threads[1]), 0] )
      
      # Number of smoothing iterations for different levels
      for x in range(1,self.Levels):
         self.LevelIter[x]=self.LevelIter[0]*pow(2,x)

      # Defining gridsteps for all levels  Number 
      for x in range(self.Levels-1,-1,-1):
         # Changed from 3 -> 2
         # NOTE: CHANGES FROM -1 to -0
         self.XStp[x]=( self.XSize/((self.XNum[x])) );
         self.YStp[x]=( self.YSize/((self.YNum[x])) );

      # Add padding
      for x in range(0,self.Levels):
         self.XNum[x] = self.XNum[x]+2
         self.YNum[x] = self.YNum[x]+2         
                        
      if (self.Equations == 2):
         self.IceTopo0 = zeros( (self.XNum[0],self.YNum[0]) )
         self.BedTopo0 = zeros( (self.XNum[0],self.YNum[0]) )



      ################################
      #       Declare levels         #
      ################################

         
      if (self.Equations == 0):
         # FINEST (PRINCIPAL) GRID
         # Defining density rho() viscosity for shear stress (etas) and 
         # viscosity for normal stress (etan) (in cells)
         # Defining initial guesses for velocity vx() vy() and pressure pr()
         # Computing right part of Stokes (RX, RY) and Continuity (RC) equation
         # vx, vy, P
         
         # Level 0
         self.Vx0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.Vy0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.P0    = zeros( (self.XNum[0],self.YNum[0]) )
         self.Rx0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.Ry0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.Rc0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.Rho0  = zeros( (self.XNum[0],self.YNum[0]) )
         self.Etan0 = zeros( (self.XNum[0],self.YNum[0]) )
         self.Etas0 = zeros( (self.XNum[0],self.YNum[0]) )
         self.Resx0 = zeros( (self.XNum[0],self.YNum[0]) )
         self.Resy0 = zeros( (self.XNum[0],self.YNum[0]) )
         self.Resc0 = zeros( (self.XNum[0],self.YNum[0]) )
         
            # Level 1
         self.Vx1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.Vy1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.P1    = zeros( (self.XNum[1],self.YNum[1]) )
         self.Rx1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.Ry1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.Rc1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.Etan1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.Etas1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.Resx1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.Resy1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.Resc1 = zeros( (self.XNum[1],self.YNum[1]) )
         
            # Level 2
         self.Vx2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.Vy2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.P2    = zeros( (self.XNum[2],self.YNum[2]) )
         self.Rx2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.Ry2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.Rc2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.Etan2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.Etas2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.Resx2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.Resy2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.Resc2 = zeros( (self.XNum[2],self.YNum[2]) )
         
         # Level 3
         self.Vx3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.Vy3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.P3    = zeros( (self.XNum[3],self.YNum[3]) )
         self.Rx3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.Ry3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.Rc3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.Etan3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.Etas3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.Resx3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.Resy3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.Resc3 = zeros( (self.XNum[3],self.YNum[3]) )
         
         # Level 4
         self.Vx4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.Vy4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.P4    = zeros( (self.XNum[4],self.YNum[4]) )
         self.Rx4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.Ry4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.Rc4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.Etan4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.Etas4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.Resx4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.Resy4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.Resc4 = zeros( (self.XNum[4],self.YNum[4]) )
         
            # Level 5
         self.Vx5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.Vy5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.P5    = zeros( (self.XNum[5],self.YNum[5]) )
         self.Rx5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.Ry5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.Rc5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.Etan5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.Etas5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.Resx5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.Resy5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.Resc5 = zeros( (self.XNum[5],self.YNum[5]) )
         
            # Level 6
         self.Vx6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.Vy6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.P6    = zeros( (self.XNum[6],self.YNum[6]) )
         self.Rx6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.Ry6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.Rc6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.Etan6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.Etas6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.Resx6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.Resy6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.Resc6 = zeros( (self.XNum[6],self.YNum[6]) )
         
            # Level 7
         self.Vx7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.Vy7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.P7    = zeros( (self.XNum[7],self.YNum[7]) )
         self.Rx7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.Ry7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.Rc7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.Etan7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.Etas7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.Resx7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.Resy7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.Resc7 = zeros( (self.XNum[7],self.YNum[7]) )
         
      if (self.Equations == 1):
         print("3D Full Stokes not implemented yet!")
         exit(0);
         
      if (self.Equations == 2):
         # Level 0
         self.Vx0      = zeros( (self.XNum[0],self.YNum[0]) )
         self.Vy0      = zeros( (self.XNum[0],self.YNum[0]) )
         self.Sxx0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Syy0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Sxy0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Exx0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Eyy0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Exy0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Exym0    = zeros( (self.XNum[0],self.YNum[0]) )
         self.Ezz0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Taue0    = zeros( (self.XNum[0],self.YNum[0]) )
         self.TaueR0   = zeros( (self.XNum[0],self.YNum[0]) )
         self.VxR0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.VyR0     = zeros( (self.XNum[0],self.YNum[0]) )
         self.Resx0    = zeros( (self.XNum[0],self.YNum[0]) )
         self.Resy0    = zeros( (self.XNum[0],self.YNum[0]) )
         self.Rest0    = zeros( (self.XNum[0],self.YNum[0]) )
         
         # Level 1
         self.IceTopo1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.BedTopo1 = zeros( (self.XNum[1],self.YNum[1]) )
         self.Vx1      = zeros( (self.XNum[1],self.YNum[1]) )
         self.Vy1      = zeros( (self.XNum[1],self.YNum[1]) )
         self.Sxx1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Syy1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Sxy1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Exx1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Eyy1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Exy1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Exym1    = zeros( (self.XNum[1],self.YNum[1]) )
         self.Ezz1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Taue1    = zeros( (self.XNum[1],self.YNum[1]) )
         self.TaueR1   = zeros( (self.XNum[1],self.YNum[1]) )
         self.VxR1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.VyR1     = zeros( (self.XNum[1],self.YNum[1]) )
         self.Resx1    = zeros( (self.XNum[1],self.YNum[1]) )
         self.Resy1    = zeros( (self.XNum[1],self.YNum[1]) )
         self.Rest1    = zeros( (self.XNum[1],self.YNum[1]) )

         # LeveL 2
         self.IceTopo2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.BedTopo2 = zeros( (self.XNum[2],self.YNum[2]) )
         self.Vx2      = zeros( (self.XNum[2],self.YNum[2]) )
         self.Vy2      = zeros( (self.XNum[2],self.YNum[2]) )
         self.Sxx2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Syy2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Sxy2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Exx2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Eyy2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Exy2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Exym2    = zeros( (self.XNum[2],self.YNum[2]) )
         self.Ezz2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Taue2    = zeros( (self.XNum[2],self.YNum[2]) )
         self.TaueR2   = zeros( (self.XNum[2],self.YNum[2]) )
         self.VxR2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.VyR2     = zeros( (self.XNum[2],self.YNum[2]) )
         self.Resx2    = zeros( (self.XNum[2],self.YNum[2]) )
         self.Resy2    = zeros( (self.XNum[2],self.YNum[2]) )
         self.Rest2    = zeros( (self.XNum[2],self.YNum[2]) )

         # LeveL 3
         self.IceTopo3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.BedTopo3 = zeros( (self.XNum[3],self.YNum[3]) )
         self.Vx3      = zeros( (self.XNum[3],self.YNum[3]) )
         self.Vy3      = zeros( (self.XNum[3],self.YNum[3]) )
         self.Sxx3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Syy3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Sxy3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Exx3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Eyy3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Exy3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Exym3    = zeros( (self.XNum[3],self.YNum[3]) )
         self.Ezz3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Taue3    = zeros( (self.XNum[3],self.YNum[3]) )
         self.TaueR3   = zeros( (self.XNum[3],self.YNum[3]) )
         self.VxR3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.VyR3     = zeros( (self.XNum[3],self.YNum[3]) )
         self.Resx3    = zeros( (self.XNum[3],self.YNum[3]) )
         self.Resy3    = zeros( (self.XNum[3],self.YNum[3]) )
         self.Rest3    = zeros( (self.XNum[3],self.YNum[3]) )

         # LeveL 4
         self.IceTopo4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.BedTopo4 = zeros( (self.XNum[4],self.YNum[4]) )
         self.Vx4      = zeros( (self.XNum[4],self.YNum[4]) )
         self.Vy4      = zeros( (self.XNum[4],self.YNum[4]) )
         self.Sxx4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Syy4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Sxy4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Exx4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Eyy4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Exy4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Exym4    = zeros( (self.XNum[4],self.YNum[4]) )
         self.Ezz4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Taue4    = zeros( (self.XNum[4],self.YNum[4]) )
         self.TaueR4   = zeros( (self.XNum[4],self.YNum[4]) )
         self.VxR4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.VyR4     = zeros( (self.XNum[4],self.YNum[4]) )
         self.Resx4    = zeros( (self.XNum[4],self.YNum[4]) )
         self.Resy4    = zeros( (self.XNum[4],self.YNum[4]) )
         self.Rest4    = zeros( (self.XNum[4],self.YNum[4]) )

         # LeveL 5
         self.IceTopo5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.BedTopo5 = zeros( (self.XNum[5],self.YNum[5]) )
         self.Vx5      = zeros( (self.XNum[5],self.YNum[5]) )
         self.Vy5      = zeros( (self.XNum[5],self.YNum[5]) )
         self.Sxx5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Syy5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Sxy5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Exx5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Eyy5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Exy5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Exym5    = zeros( (self.XNum[5],self.YNum[5]) )
         self.Ezz5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Taue5    = zeros( (self.XNum[5],self.YNum[5]) )
         self.TaueR5   = zeros( (self.XNum[5],self.YNum[5]) )
         self.VxR5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.VyR5     = zeros( (self.XNum[5],self.YNum[5]) )
         self.Resx5    = zeros( (self.XNum[5],self.YNum[5]) )
         self.Resy5    = zeros( (self.XNum[5],self.YNum[5]) )
         self.Rest5    = zeros( (self.XNum[5],self.YNum[5]) )

         # LeveL 6
         self.IceTopo6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.BedTopo6 = zeros( (self.XNum[6],self.YNum[6]) )
         self.Vx6      = zeros( (self.XNum[6],self.YNum[6]) )
         self.Vy6      = zeros( (self.XNum[6],self.YNum[6]) )
         self.Sxx6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Syy6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Sxy6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Exx6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Eyy6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Exy6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Exym6    = zeros( (self.XNum[6],self.YNum[6]) )
         self.Ezz6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Taue6    = zeros( (self.XNum[6],self.YNum[6]) )
         self.TaueR6   = zeros( (self.XNum[6],self.YNum[6]) )
         self.VxR6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.VyR6     = zeros( (self.XNum[6],self.YNum[6]) )
         self.Resx6    = zeros( (self.XNum[6],self.YNum[6]) )
         self.Resy6    = zeros( (self.XNum[6],self.YNum[6]) )
         self.Rest6    = zeros( (self.XNum[6],self.YNum[6]) )

         # LeveL 7
         self.IceTopo7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.BedTopo7 = zeros( (self.XNum[7],self.YNum[7]) )
         self.Vx7      = zeros( (self.XNum[7],self.YNum[7]) )
         self.Vy7      = zeros( (self.XNum[7],self.YNum[7]) )
         self.Sxx7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Syy7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Sxy7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Exx7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Eyy7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Exy7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Exym7    = zeros( (self.XNum[7],self.YNum[7]) )
         self.Ezz7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Taue7    = zeros( (self.XNum[7],self.YNum[7]) )
         self.TaueR7   = zeros( (self.XNum[7],self.YNum[7]) )
         self.VxR7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.VyR7     = zeros( (self.XNum[7],self.YNum[7]) )
         self.Resx7    = zeros( (self.XNum[7],self.YNum[7]) )
         self.Resy7    = zeros( (self.XNum[7],self.YNum[7]) )
         self.Rest7    = zeros( (self.XNum[7],self.YNum[7]) )
