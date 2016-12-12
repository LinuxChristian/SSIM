/**/

#ifndef CONFIG_H_
#include "SSIM/src/config.h"
#endif


void setPointers (Multigrid &MG, ValueType *Vx, ValueType *Vy, ValueType *Vxs, ValueType *Vys,
		  ValueType *Vxb, ValueType *Vyb,
		  ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue,
		  ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
		  ValueType *IceTopo, ValueType *BedTopo, ValueType *Pw,
		  ValueType *tbx, ValueType *tby, ValueType *tbz, 
		  ValueType *ts, ValueType *tn, ValueType *vb, ValueType *beta,
		  ValueType *dhdx, ValueType *dhdy, 
		  ValueType *dbdx, ValueType *dbdy,
		  ValueType *Scx, ValueType *Scy, ValueType *hw, 
		  ValueType *qsx, ValueType *qsy,
		  ValueType *Qcx, ValueType *Qcy, ValueType *R, 
		  ValueType *Psi, ValueType *Mcx, ValueType *Mcy ) {

  MG.FixSPMPointers(Vx, Vy, Vxs, Vys, Vxb, Vyb, 
		    Sxx, Syy, Sxy, Szz, Taue, 
		    Exx, Eyy, Exy, Ezz, 
		    IceTopo, BedTopo, 
		    Pw, tbx, tby, tbz, ts, tn, vb, beta,
		    dhdx, dhdy, dbdx, dbdy,
		    Scx, Scy, hw, qsx, qsy,
		    Qcx, Qcy, R, Psi, Mcx, Mcy);
}

// Copy gradients from GPU to CPU
void unRollGradientStructures (hptype** hp, vptype **vp, meshtype &mesh,
			     ValueType *dbdx, ValueType *dbdy,
			     ValueType *dhdx, ValueType *dhdy) {

  int XNum = mesh.nx;
  int YNum = mesh.ny;

  /* x direction */
    for (int y=0;y<YNum;y++) {
      for (int x=1;x<XNum;x++) {
	(dhdx)[(y+1)*(XNum+2)+(x+1)] = hp[y][x].dhdx;
	(dbdx)[(y+1)*(XNum+2)+(x+1)] = hp[y][x].dbdx;
      }
    }

  /* y direction */
    for (int y=1;y<YNum;y++) {
      for (int x=0;x<XNum;x++) {
	(dhdy)[(y+1)*(XNum+2)+(x+1)] = vp[y][x].dhdy;
	(dbdy)[(y+1)*(XNum+2)+(x+1)] = vp[y][x].dbdy;
      }
    }

}

void unrollStructures(celltype** cells,hptype** hp,
		      vptype** vp, cornertype** cp, meshtype &mesh, 
		      ValueType *IceTopo, ValueType *BedTopo, 
		      ValueType *dhdx, ValueType *dhdy, 
		      ValueType *dbdx, ValueType *dbdy, 
		      ValueType *Pw, ValueType *R,
		      double a2w) {


  
  int XNum = mesh.nx;
  int YNum = mesh.ny;

  // Fix topo BC's
  for (int y=0;y<YNum;y++) {
    for (int x=0;x<XNum;x++) {
      (IceTopo)[(y)*(XNum+2)+(x)]  = 0.0;
      (BedTopo)[(y)*(XNum+2)+(x)]  = 0.0;
    }
  }  

  for (int y=0;y<YNum;y++) {
    for (int x=0;x<XNum;x++) {
      (IceTopo)[(y+1)*(XNum+2)+(x+1)]  = cells[y][x].topice;
      (BedTopo)[(y+1)*(XNum+2)+(x+1)]  = cells[y][x].bed;

      Pw[(y+1)*(XNum+2)+(x+1)]      = cells[y][x].Pw;
      R[(y+1)*(XNum+2)+(x+1)] = (cells[y][x].meltrate)*3.171e-8*0.1; // m/s

    };
  };

  unRollGradientStructures (hp, vp, mesh,
			    dbdx, dbdy,
			    dhdx, dhdy);
}


// Copy gradients from GPU to CPU
void rollGradientStructures (hptype** hp, vptype **vp, meshtype &mesh,
			     ValueType *dbdx, ValueType *dbdy,
			     ValueType *dhdx, ValueType *dhdy) {

  int XNum = mesh.nx;
  int YNum = mesh.ny;

    for (int y=0;y<YNum;y++) {
      for (int x=0;x<XNum;x++) {
	hp[y][x].dhdx      = (dhdx)[(y+1)*(XNum+2)+(x+2)];
	hp[y][x].dbdx      = (dbdx)[(y+1)*(XNum+2)+(x+2)];
      }
    }
    
    for (int y=0;y<YNum;y++) {
      for (int x=0;x<XNum;x++) {
	vp[y][x].dhdy      = (dhdy)[(y+2)*(XNum+2)+(x+1)];
	vp[y][x].dbdy      = (dbdy)[(y+2)*(XNum+2)+(x+1)];
      }
    }

}

void rollStructures (celltype** cells,hptype** hp,vptype** vp, cornertype** cp, meshtype &mesh, 
		     ValueType *Vx, ValueType *Vy, ValueType *Vxs, ValueType *Vys,		    
		     //		     ValueType *Vxd, ValueType *Vyd,
		     ValueType *Vxb, ValueType *Vyb,
		     ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Taue,
		     ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
		     ValueType *IceTopo, ValueType *BedTopo, ValueType *Pw,
		     ValueType *Ts, ValueType *Tn, ValueType *Vb, ValueType *beta, 
		     ValueType *Tbx, ValueType *Tby, ValueType *Tbz,
		     ValueType *dhdx, ValueType* dhdy, ValueType* dbdx, ValueType* dbdy,
		     ValueType *Scx, ValueType *Scy, ValueType * hw,
		     const bool fileio) {
  int XNum = mesh.nx;
  int YNum = mesh.ny;
  
  // update hp
  for (int y=0;y<YNum;y++) {
    for (int x=1;x<=XNum;x++) {   
      hp[y][x].vx = Vx[(y+1)*(XNum+2)+(x+1)]; // Array is reused! This is not a typo!
      hp[y][x].vx_s = Vxs[(y+1)*(XNum+2)+(x+1)];
      hp[y][x].vx_b = Vxb[(y+1)*(XNum+2)+(x+1)];
      //      hp[y][x].vx = hp[y][x].vx_d+hp[y][x].vx_b;
      hp[y][x].vx_d = hp[y][x].vx-hp[y][x].vx_b;
    }
  }
  
  for (int y=1;y<=YNum;y++) {
    for (int x=0;x<XNum;x++) {
      vp[y][x].vy   = Vy[(y+1)*(XNum+2)+(x+1)]; // Array is reused! This is not a typo!
      vp[y][x].vy_s = Vys[(y+1)*(XNum+2)+(x+1)];
      vp[y][x].vy_b = Vyb[(y+1)*(XNum+2)+(x+1)];
      //      vp[y][x].vy = vp[y][x].vy_d+vp[y][x].vy_b;
      vp[y][x].vy_d = vp[y][x].vy-vp[y][x].vy_b;
    }
  }
  
    for (int y=0;y<YNum;y++) {
      for (int x=0;x<XNum;x++) {
	cells[y][x].tn   = Tn[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].ts   = Ts[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].te   = cells[y][x].tn - cells[y][x].Pw;

	//	cells[y][x].dbdx   = dbdx[(y+1)*(XNum+2)+(x+1)];
	//	cells[y][x].dbdy   = dbdy[(y+1)*(XNum+2)+(x+1)];
      }
    }

  if (fileio) {
    // Only update cells if writing file
    // in this timestep
    for (int y=0;y<YNum;y++) {
      for (int x=0;x<XNum;x++) {
	// Velocity     
	cells[y][x].sxx = Sxx[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].syy = Syy[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].sxy = Sxy[(y+1)*(XNum+2)+(x+1)];
	
	cells[y][x].exx = Exx[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].eyy = Eyy[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].exy = Exy[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].ezz = Ezz[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].te2 = Taue[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].beta = beta[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].tn   = Tn[(y+1)*(XNum+2)+(x+1)];
	cells[y][x].ts   = Ts[(y+1)*(XNum+2)+(x+1)];

	if (mesh.dotestA){	  
	  cells[y][x].topice    = (IceTopo)[(y+1)*(XNum+2)+(x+1)];
	  cells[y][x].bed       = (BedTopo)[(y+1)*(XNum+2)+(x+1)];
	};
      };
    };
  };

}

void cpyMemToGPU (Multigrid &MG) {

  // Note: This should only copy the first level to device!
  MG.CopyLevelToDevice(0);
  return;
}

void cpyGPUToMem (Multigrid &MG, const bool FileIO) {

  MG.CopyLevelToHost(0, FileIO);
  return;
}

void GPUComputeGradients (Multigrid &MG) {
  
  for (unsigned int i = 0; i < MG.GetCurrentLevels(); i++) {
      MG.ComputeSurfaceGradients(i);
  }
 
  
}


Multigrid initializeMG (char *path,char file[200], 
			int xdim, int ydim, int xnum, int ynum,
			ValueType L0, ValueType minbeta, 
			ValueType C, ValueType Cs,
			ValueType maxsliding,
			ValueType vbfac, bool dotestA,
			ValueType *Vx, ValueType *Vy, ValueType *Vxs, ValueType *Vys,
			ValueType *Vxb, ValueType *Vyb,
			ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue,
			ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
			ValueType *IceTopo, ValueType *BedTopo, 
			ValueType *dhdx, ValueType *dhdy, 
			ValueType *dbdx, ValueType *dbdy,
			ValueType *Pw,
			ValueType *tbx, ValueType *tby, ValueType *tbz, 
			ValueType *ts, ValueType *tn, ValueType *vb, ValueType *beta,
			ValueType *Scx, ValueType *Scy, ValueType *hw, 
			ValueType *qsx, ValueType *qsy,
			ValueType *Qcx, ValueType *Qcy, ValueType *R, 
			ValueType *Psi, ValueType *Mcx, ValueType *Mcy,
			mgproptype &mgprop, meshtype &mesh, 
			iproptype &iprop, mproptype &mprop
			) {

  int CPU = 0;

  // Make sure MG is active
  mgprop.mg_active = 1;

  size_t TotalMemSize = sizeof(ValueType)*10000*10000; // Size of avaible memory in bytes
                                                       // Only important on GPU

  size_t TotalConstMemSize = sizeof(ValueType)*10000*10000;

  Multigrid MG(CPU, TotalMemSize, TotalConstMemSize, mgprop.SingleStp, mgprop.StpLen, mgprop.FirstStp,
	       mgprop.Converg_s, mgprop.Converg_v);
  MG.initializeGPU(&TotalMemSize, &TotalConstMemSize);
  printf("GPU Initalized\n");
  checkForCudaErrors("Initilazation of GPU!",0);

  printf("%i %i %i", mgprop.requestedLevels, mgprop.threadsPerBlock[0], mgprop.threadsPerBlock[1]);
  
  MG.TransferSettingsToSSIM(mgprop.requestedLevels, mgprop.restrictOperator, mgprop.prolongOperator, 
			    mgprop.iterationsOnLevel, mgprop.equations, mgprop.startLevel, iprop.maxitt_v,
			    xnum, ynum, mesh.L, mesh.H, 
			    mgprop.bc, iprop.sfac, iprop.ifac, iprop.gamma, iprop.gamma0,
			    mprop.mtype, mprop.lrate, mprop.T0, mprop.maxacc, mprop.accgrad,
			    mprop.ablgrad, mprop.maxabla, iprop.latentheat, mprop.avaslope, mprop.maxslope,
			    mesh.gravity,
			    mgprop.vKoef, mgprop.sKoef,
			    mesh.maxs, mesh.maxb,
			    1000.0,917.0, 
			    0.05, // Sheet conductivity
			    0.1, // turbulent flow coefficient
			    0.001,  // Relaxation
			    20,  // incipient channel width
			    1,  // Critical water depth
			    1, // expoent (water depth)
			    0.063, // Geothermal heat
			    0.5,
			    5,
			    mgprop.threadsPerBlock, mgprop.blocksPerGrid,
			    L0, minbeta, C, Cs,
			    maxsliding, vbfac,
			    mesh.dosliding, mesh.slidingmode);
  
  MG.DisplaySetup();

  printf("Setting Constant Memory on Device\n");
  MG.SetConstMemOnDevice();
  checkForCudaErrors("Setting constant memory!",0);
  
  // Allocate all memory needed for levels
  // This is done on both GPU and in RAM
  MG.AddLevels(iprop.maxitt_s);

  printf("Setting new pointer location\n");
  fflush(stdout);

  setPointers(MG, Vx, Vy, Vxs, Vys, Vxb, Vyb,
	      Sxx, Syy, Sxy, Szz, Taue, 
	      Exx, Eyy, Exy, Ezz, 
	      IceTopo, BedTopo, Pw,
	      tbx, tby, tbz, ts, tn, vb, beta,
	      dhdx, dhdy, dbdx, dbdy,
	      Scx, Scy, hw, qsx, qsy,
	      Qcx, Qcy, R, Psi, Mcx, Mcy);
  checkForCudaErrors("Could not set new pointers in interface", 1);

  printf("\n%p\n", Vx);

  return MG;
}

void GPUHydrology(Multigrid &MG, double dt) 
		  /*		  ,ValueType *Scx, ValueType *Scy, ValueType *hw, 
		  ValueType *qsx, ValueType *qsy,
		  ValueType *Qcx, ValueType *Qcy, ValueType *R, 
		  ValueType *Psi, ValueType *Mcx, ValueType *Mcy )  
		  */
		  {
  MG.SolveHydrology(dt);
}

//void iSOSIA(Multigrid &MG) {
void iSOSIA(Multigrid &MG, celltype** cells,hptype** hp,vptype** vp, cornertype** cp, meshtype &mesh, mgproptype &mgprop, hwproptype &hwprop,
	    double time,
	    ValueType *Vx, ValueType *Vy, ValueType *Vxs, ValueType *Vys,
	    ValueType *Vxb, ValueType *Vyb,
	    //	    ValueType *Vxd, ValueType *Vyd,
	    ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue,
	    ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
	    ValueType *IceTopo, ValueType *BedTopo, 
	    ValueType *dhdx, ValueType *dhdy, 
	    ValueType *dbdx, ValueType *dbdy,
	    ValueType *Pw,
	    ValueType *tbx, ValueType *tby, ValueType *tbz, 
	    ValueType *ts, ValueType *tn, ValueType *vb, ValueType *beta, 
	    ValueType *Scx, ValueType *Scy, ValueType * hw, ValueType *R,
	    const double dt, const int maxittv, const int maxitts,
	    const bool FileIO,
	    const int ForceVelocity
	    ) {
  
  // Send everything to SSIM
  if (mesh.dotestA) {
    MG.setTestATopo();
    
      for (unsigned int level=0;level<MG.GetCurrentLevels()-1;level++) {
	MG.RestrictTopography(level, level+1, 0);
      };
      
      GPUComputeGradients(MG);

  } else {
    unrollStructures(cells, hp, vp, cp, mesh, IceTopo, BedTopo, dhdx, dhdy, dbdx, dbdy, Pw, R,hwprop.a2w);
    
    
    if (mgprop.mg_active) {

      for (unsigned int level=0;level<MG.GetCurrentLevels()-1;level++) {
	MG.RestrictTopography(level, level+1, 0);
      };
      
      //  GPUComputeGradients(MG);
    } else {
      // MG Disabled!
      MG.ComputeSurfaceGradients(0);
    };
    
    cpyMemToGPU(MG);
  };

  std::cout << std::endl;

  printf("LOG:","------ Starting Multigrid ------");
  
 // Get initial guess on solution
 // MG.RunFMG(prof);
 
 MG.SetCycleNum(-1);
 
 // Reset first level
 MG.SetFirstLevel( );
 int NextResUpdate = 1; // Get a velocity residual to begin with
 int LastResUpdate = -1;
 int ExpectedConvergence = 0;
 ValueType PreviousResidual = 0.0;
 ValueType SlidingError = 1e20;
 profile prof; // Create variable for profiling
 bool doMG = (bool) mgprop.mg_active;
 double grad=0.0;

 if (doMG) {
   printf("MG Enabled \n");
 } else {
   printf("MG Disabled \n");
 };
 (mesh).stressitt = 0;
  
 // Compute hydrology on GPU
 // One day time steps
 /*
 int ndtw = dt*365*60*60*24/(60*60);
 printf("Solving hydrlogy - dt %f ndtw %i\n",dt,ndtw);
 for (int i=0;i<ndtw;i++) GPUHydrology(MG, dt*(60*60*24*365)/((double)ndtw));
 */
 fflush(stdout);
 ValueType VelocityError = 1e2;
 while (MG.GetCycleNum() < maxittv && 
	(
	 ( VelocityError > 1e-3 || SlidingError > 1e-3 ) 
	 //	 ||  (VelocityError < 1e-30 || SlidingError > 1e-30) 
	 ) 
	)
    {
      // std::cout << "\r Running cycle " << MG.GetCycleNum() << " Current residual " << VelocityError << " Next update will be "<< NextResUpdate << " Convergence expected "<<  ExpectedConvergence;


      (mesh).stressitt +=  MG.RunVCycle(prof,doMG,ForceVelocity);

      // MG.RunWCycle(prof);
      checkForCudaErrors("Done with cycle", 1);
      /*
      std::cout << "Next res update will be: " << NextResUpdate << std::endl;
      std::cout << "Convergence is expected at: " << ExpectedConvergence << std::endl;
      */

      // The velocity residual is only computed if need be!
      // This is a very slow function!
      if ( NextResUpdate == MG.GetCycleNum() ||
	   MG.GetCycleNum() == ExpectedConvergence || 
	   1) {

	// Update velocity residual from GPU
	VelocityError = MG.VelocityResidual(prof);

	grad = ((log(VelocityError/PreviousResidual)))
	  /(MG.GetCycleNum() - LastResUpdate); // grad = (y2 - y1)/(x2 - x1)	


	MG.ResidualCheck(MG.GetCycleNum(), VelocityError, 
			 NextResUpdate, LastResUpdate, PreviousResidual, ExpectedConvergence,
			 mgprop.SingleStp, mgprop.StpLen, mgprop.FirstStp, mgprop.Converg_v);

	MG.setResidual(VelocityError, MG.GetCycleNum());

	//	printf("Velocity residual: %f Convergence factor: %f\n",VelocityError,grad);
	//	fflush(stdout);
	
	if (mgprop.autoControl && VelocityError < mgprop.disableCriteria) {
	  doMG = false;
	} else {
	  doMG = true;
	};
	if (VelocityError == 0.0)
	  break;
      }
      
      if (VelocityError > 1e4) {
	std::cout << "\n----------------------------\n";
	std::cout << "Did not converge! \n";
	std::cout << "----------------------------\n\n";
	break;
      };

      /* Try and reduce relaxation if residual is growing */
      if ( ( MG.getResidual(MG.GetCycleNum()-1) - MG.getResidual(MG.GetCycleNum()) ) < -1e-3 && 0 ) {
	MG.ReduceRelaxation(0.5);
      };

      SlidingError = MG.SlidingResidual();
      //      printf("Sliding Residual is: %f \n",SlidingError);
      MG.UpdateCycleNum();

    };

 printf("Done with V-cycle\n");
 (mesh).veloitt = MG.GetCycleNum();
 (mesh).veloitt_count += 1;
 
 if (doMG) {
   mgprop.mg_active = 1;
 } else {
   mgprop.mg_active = 0;
 };

 // MG.ComputeSliding();

 // Do a full update of surface velocities
 MG.ComputeSurfaceVelocity(2);

 VelocityError = MG.VelocityResidual(prof);
  
 MG.ResidualCheck(MG.GetCycleNum(), VelocityError, 
		  NextResUpdate, LastResUpdate, 
		  PreviousResidual, ExpectedConvergence,
		  mgprop.SingleStp, mgprop.StpLen, mgprop.FirstStp, mgprop.Converg_v);

  std::cout << "Final Stress iterations: " << (mesh).stressitt  << std::endl;
  std::cout << "Final Velocity Residual: " << VelocityError << " with " <<  MG.GetCycleNum() << " iterations" << std::endl;

  // Get everything ready for SPM
  cpyGPUToMem(MG, FileIO);	

  printf("Rolling structure\n");
  rollStructures(cells, hp, vp, cp, mesh,
		 Vx,  Vy,  Vxs,  Vys,
		 //		 Vxd, Vyd,
		 Vxb, Vyb,
		 Sxx,  Syy,  Sxy,  Taue,
		 Exx,  Eyy,  Exy,  Ezz, 
		 IceTopo,  BedTopo, Pw, 
		 ts, tn, vb, beta,
		 tbx, tby, tbz,
		 dhdx, dhdy, dbdx, dbdy,
		 Scx, Scy, hw, 
		 FileIO);

    
}
