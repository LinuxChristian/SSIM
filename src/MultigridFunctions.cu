
#ifndef CONFIG_H_
#include "config.h"
#include <iostream>
#include <fstream>
#endif

#ifdef __GPU__
#define DEVICESPECS <<<grid_size, block_size>>>
#endif

Multigrid::Multigrid( const int CPU, const size_t TotalMemSize, const size_t TotalConstMemSize,
		     int SingleStp, double StpLen, int FirstStp, double Converg_s, double Converg_v) 
{  
  if (&MGS == NULL)
    Error(-6,"Could not allocate MGS in Multigrid construct");

  MGS.CPU = CPU;
  MGS.TotalMemSize = TotalMemSize;
  MGS.TotalConstMemSize = TotalConstMemSize;

  // Settings for convergence computations
  MGS.FirstStp  = FirstStp;      // Number of iterations before first convergence test
  MGS.StpLen          = StpLen;       
  MGS.SingleStp       = SingleStp; 
  MGS.Converg_s       = Converg_s;
  MGS.Converg_v       = Converg_v;
};

/*! 
  Read solver configurations from a binary file.
  
  If not used this function should be deleted
 */
void Multigrid::TransferSettingsToSSIM(int requestedLevels, int restrictOperator, int prolongOperator,
				       int *iterationsOnLevel, int equations, int startLevel, int cycles,
				       int xnum, int ynum, int L, int H, 
				       int *bc, double sRelax, double vRelax, double gamma, double gamma0,
				       int mtype, double lrate, double T0, double maxacc, double accgrad,
				       double ablagrad, double maxabla, double latentheat, double avaslope, double maxslope,
				       double g,
				       double  vKoef, double sKoef,
				       double maxs, double maxb,
				       double rho_w, double rho_i,
				       double ks, double kc, 
				       double Relaxh, double lambdac,
				       double hc, double gamma_h,
				       double GeoFlux,
				       double hr, double lr,
				       int *threadsPerBlock, int *blocksPerGrid,
				       // Sliding Variables
				       ValueType L0, ValueType minbeta, ValueType C, 
				       ValueType Cs,
				       ValueType maxSliding, ValueType vbfac,
				       int dosliding, int slidingmode
				       ) {

  MGS.totIterNum = 0;  // Total number of iterations
  MGS.Direction = 1;   // DEFAULT: Restriction
  MGS.startLevel = 0;  // DEFAULT: 0 (ONLY used for debugging!)
  MGS.IOUpdate = 0;
  MGS.startIteration = 0;

  // Get general values from the NetCDF file
  MGS.RequestedLevels = requestedLevels;
  MGS.IOUpdate = 1;
  MGS.RestrictOperator = restrictOperator;
  MGS.ProlongOperator = prolongOperator;
  for (int i=0; i<MGS.RequestedLevels; i++) {
    MGS.IterationsOnLevel[i] = iterationsOnLevel[i];
    ConstMem.IterNum[i] = 1; 
  }

  MGS.equations = equations;
  MGS.startLevel = startLevel;

  ConstMem.XNum[0] = xnum;
  ConstMem.YNum[0] = ynum;
  ConstMem.XDim = L;
  ConstMem.YDim = H; 

  // Compute dimensions of coarse grids
  for (int i=1; i<MGS.RequestedLevels; i++) {
    ConstMem.XNum[i] = floor((ConstMem.XNum[0]-1)/pow(2,i)+1);
    ConstMem.YNum[i] = floor((ConstMem.YNum[0]-1)/pow(2,i)+1);
  }

  for (int i=0; i<MGS.RequestedLevels; i++) {
    ConstMem.Dx[i] = ConstMem.XDim/((ConstMem.XNum[i])); 
    ConstMem.Dy[i] = ConstMem.YDim/((ConstMem.YNum[i])); 

    ConstMem.XNum[i]+=2; // Add padding
    ConstMem.YNum[i]+=2;
  }
  
  for (int i=0; i<4; i++)
    ConstMem.BC[i] = bc[i];
 
  MGS.NumberOfCycles = cycles;
  
 if (MGS.startIteration > 1) {
   // Allow more iterations then normally set
   MGS.NumberOfCycles += MGS.startIteration;
 };

 // Constant Memory
 ConstMem.Relaxs = sRelax;
 ConstMem.Relaxv = vRelax;
 ConstMem.Gamma = gamma0;
 ConstMem.A = gamma;
 ConstMem.vKoef = vKoef;
 ConstMem.sKoef = sKoef;
 ConstMem.maxb = maxb;
 ConstMem.maxs = maxs;

 ConstMem.mtype = mtype;
 ConstMem.lrate = lrate;
 ConstMem.T0 = T0;
 ConstMem.maxacc = maxacc;
 ConstMem.accgrad = accgrad;
 ConstMem.avaslope = avaslope;
 ConstMem.maxslope = maxslope;
 ConstMem.latentheat = latentheat;
 ConstMem.maxabla = maxabla;
 ConstMem.g = g;
 
 // Sliding variables
 ConstMem.L0 = L0;
 ConstMem.minbeta = minbeta;
 ConstMem.C = C;
 ConstMem.Cs = Cs;
 ConstMem.maxsliding = maxSliding;
 ConstMem.Relaxvb = vbfac;   
 
 ConstMem.rho_w =  rho_w;
 ConstMem.rho_i = rho_i;
 ConstMem.ks = ks; // Sheet hydralic conductivity
 ConstMem.kc = kc; // Channel turbulent flow coefficient
 ConstMem.Relaxh = Relaxh; // Relax for hydrology
 ConstMem.lambdac = lambdac; // incipient channel width
 ConstMem.hc = hc; // critical layer depth
 ConstMem.gamma_h = gamma_h; 
 ConstMem.GeoFlux = GeoFlux;
 ConstMem.hr = hr;
 ConstMem.lr = lr;

 ConstMem.dosliding = dosliding;
 ConstMem.slidingmode = slidingmode;
 // Check the number of levels
  // Max 0 - 7 levels
  if (MGS.RequestedLevels > maxLevel) {
    Error(-1,"Cannot handle that many levels!");
  };
 
  /* Read GPU setup */
  for (int i=0; i<3; i++) {
    MGS.threadsPerBlock[i] = threadsPerBlock[i];
    MGS.numBlock[i] = blocksPerGrid[i];
  }
  

  printf("Threads %i %i\n",threadsPerBlock[0], threadsPerBlock[1]);

 /////////////////////////////////////////////////////////
 //                   CHECK INPUT                       //
 /////////////////////////////////////////////////////////

 if ( ( (ConstMem.BC[0] == 2 || ConstMem.BC[1] == 2 ) && ConstMem.BC[0] != ConstMem.BC[1]) || ( (ConstMem.BC[2] == 2 || ConstMem.BC[3] == 2) && ConstMem.BC[2] != ConstMem.BC[3]) ) {
   Error(-1,"Periodic boundaries needs to be applied to both sides");
 };

};

void Multigrid::ComputeSurfaceGradients(const int Level) {
  Levels[Level]->getGradients();
};

void Multigrid::initializeGPU(size_t* TotalGlobalMem, size_t* TotalConstMem)
{
#ifdef __GPU__
  // Specify target device
  int cudadevice = 0;
  
  // Variables containing device properties
  cudaDeviceProp prop;
  int devicecount;
  int cudaDriverVersion;
  int cudaRuntimeVersion;
  
  
  // Register number of devices
  cudaGetDeviceCount(&devicecount);
  checkForCudaErrors("Initializing GPU!");

  if(devicecount == 0) {
    printf("\nERROR:","No CUDA-enabled devices availible. Bye.\n");
    exit(EXIT_FAILURE);
  } else if (devicecount == 1) {
    printf("\nSystem contains 1 CUDA compatible device.\n","");
  } else {
    printf("\nSystem contains %i CUDA compatible devices.\n",devicecount);
  }
  
  cudaGetDeviceProperties(&prop, cudadevice);
  cudaDriverGetVersion(&cudaDriverVersion);
  cudaRuntimeGetVersion(&cudaRuntimeVersion);
  checkForCudaErrors("Initializing GPU!");
  
  // Comment following line when using a system only containing exclusive mode GPUs
  cudaChooseDevice(&cudadevice, &prop);
  checkForCudaErrors("Initializing GPU!");
  (*TotalGlobalMem) = prop.totalGlobalMem;
  (*TotalConstMem) = prop.totalConstMem;

  printf("Using CUDA device ID: %i \n",(cudadevice));
  printf("  - Name: %s, compute capability: %i.%i.\n",prop.name, prop.major, prop.minor);
  printf("  - Global Memory: %f, Constant Memory: %f.\n",prop.totalGlobalMem, prop.totalConstMem);

#else
    printf( "Code is not compiled for GPU!");
    (*TotalGlobalMem) = 0.0;
    (*TotalConstMem) = 0.0; 
#endif

};

void Multigrid::ComputeSliding() {
    Levels[0]->ComputeSlidingVelocity();
};

void Multigrid::FixSPMPointers(ValueType *Vx, ValueType *Vy, ValueType *Vxs, ValueType *Vys,
			       ValueType *Vxb, ValueType *Vyb,
			       ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue,
			       ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz,
			       ValueType *IceTopo, ValueType *BedTopo, ValueType *Pw,
			       ValueType *tbx, ValueType *tby, ValueType *tbz, 
			       ValueType *ts, ValueType *tn, ValueType *vb, ValueType *beta,
			       ValueType *dhdx, ValueType *dhdy, ValueType *dbdx, ValueType *dbdy,
			       ValueType *Scx, ValueType *Scy, ValueType *hw, ValueType *qsx, ValueType *qsy,
			       ValueType *Qcx, ValueType *Qcy, ValueType *R, ValueType *Psi, ValueType *Mcx, ValueType *Mcy) {


  Levels[0]->FixSPMPointers(Vx, Vy, Vxs, Vys, Vxb, Vyb, 
			    Sxx, Syy, Sxy, Szz, Taue, 
			    Exx, Eyy, Exy, Ezz, 
			    IceTopo, BedTopo, 
			    Pw, tbx, tby, tbz, ts, tn, vb, beta,
			    dhdx, dhdy, dbdx, dbdy,
			    Scx, Scy, hw, qsx, qsy,
			    Qcx, Qcy, R, Psi, Mcx, Mcy);
};

void Multigrid::setTestATopo() {
  for (int i=0;i<1;i++) { 
    printf("Setting topography on level %i\n",i);
    Levels[i]->setTestATopo(ConstMem.Dx[i], ConstMem.Dy[i], ConstMem.XDim, ConstMem.YDim);
  };

};

void Multigrid::DisplaySetup() {

  std::cout << "Multigrid Init()" << std::endl;

  /* Check that there is room on the GPU for the problem */
  size_t RequestedMemSize = 0;
  int ArraysPrLevel = 18; // Only valid for iSOSIA case!

  for (int i=0; i<=MGS.RequestedLevels;i++) {
    RequestedMemSize += sizeof(ValueType)*ConstMem.XNum[i]*ConstMem.YNum[i]*ArraysPrLevel; // Gridsize of level * Number Of Grids pr. level
  };

  if (MGS.TotalMemSize < RequestedMemSize) {
    std::cout << "You have requested more memory then the system has." << "\n";
    std::cout << "Requested domain size is " << ConstMem.XNum[0] << "x" << ConstMem.YNum[0] << "\n";
    std::cout << "There is " << MGS.TotalMemSize/pow(1000,2) << " MB on the system," << "\n";
    std::cout << "You have requested " << RequestedMemSize/pow(1000,2) << " MB" << "\n";
    std::cout << "Please buy more memory or reduce the size of your problem" << std::endl;
    //    exit(0);
  };

  /* Check shared memory requirements before proceeding */
  if (MGS.TotalConstMemSize < sizeof(ConstMem)) {
    std::cout << "You have requested more constant memory then the system has." << "\n";
    std::cout << "Requested domain size is " << ConstMem.XNum[0] << "x" << ConstMem.YNum[0] << "\n";
    std::cout << "There is " << MGS.TotalConstMemSize/pow(1000,1) << " KB on the system," << "\n";
    std::cout << "You have requested " << sizeof(ConstMem)/pow(1000,1) << " KB" << "\n";
    std::cout << "Please buy a biggere GPU card or reduce the size of your problem" << std::endl;
    //    exit(0);
  };

  if (1)
    {
    std::cout << "Allocated the following GPU Memory:" << "\n";
    std::cout << "Constant Memory:    " << sizeof(ConstMem)/pow(1000,1) << "/" << MGS.TotalConstMemSize/pow(1000,1) << " KB" << "\n";
    std::cout << "Global Memory:  " << RequestedMemSize/pow(1000,2) << "/" << MGS.TotalMemSize/pow(1000,2) << " MB" << "\n\n";
    };


  /* TODO: Move to seperate logging function */
   fprintf(stdout,"INITIAL SETUP\n");
   // Print 2D Stokes specific variables
   #ifdef __2DSTOKES__
   fprintf(stdout,"Relaxs %.2f Relaxc %.2f \n",ConstMem.Relaxs, ConstMem.Relaxc);
   fprintf(stdout,"vKoef %.2f pKoef %.2f \n",ConstMem.vKoef,ConstMem.pKoef);
   fprintf(stdout,"pnorm %.2f \n",ConstMem.pnorm[0]);
   #endif

   // Print iSOSIA specific variables
   #ifdef __iSOSIA__
   fprintf(stdout,"Relaxs %.2e Relaxv %.2e \n",ConstMem.Relaxs, ConstMem.Relaxv);
   fprintf(stdout,"vKoef %.2e sKoef %.2e \n",ConstMem.vKoef,ConstMem.sKoef);
   fprintf(stdout,"Gamma %.2e A(T) %.2e \n",ConstMem.Gamma,ConstMem.A);
   #endif

   fprintf(stdout,"XDim %.2f YDim %.2f \n",ConstMem.XDim,ConstMem.YDim);
   fprintf(stdout,"Number of RequestedLevels %i \n",MGS.RequestedLevels);
   fprintf(stdout,"Multigrid iterations %i \n",MGS.NumberOfCycles);
   fprintf(stdout,"Output Updates %i \n",MGS.IOUpdate);
   fprintf(stdout,"Start Level %i \n",MGS.startLevel);
   fprintf(stdout,"startIteration %i \n", MGS.startIteration);
   fprintf(stdout,"Level Iterations ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout," %i ",MGS.IterationsOnLevel[i]);
   };
   fprintf(stdout,"\nIterations between syncs ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout," %i ",ConstMem.IterNum[i]);
   };

   // Set Grid and Block Size
   MGS.GPUBlockSize.x = MGS.threadsPerBlock[0];
   MGS.GPUBlockSize.y = MGS.threadsPerBlock[1];
   MGS.GPUBlockSize.z = 1;
   MGS.GPUGridSize.x = MGS.numBlock[0];
   MGS.GPUGridSize.y = MGS.numBlock[1];   
   MGS.GPUGridSize.z= 1;   

   fprintf(stdout,"\nBlock Sizes ");
   fprintf(stdout,"[ %i %i %i ] ", MGS.threadsPerBlock[0], MGS.threadsPerBlock[1], MGS.threadsPerBlock[2]);

   fprintf(stdout,"\nGrid Sizes ");
   fprintf(stdout," [ %i %i %i ] ", MGS.numBlock[0], MGS.numBlock[1], MGS.numBlock[2]);

   fprintf(stdout,"\nxNum (w/ padding) ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout," %i ",ConstMem.XNum[i]);
   };
   fprintf(stdout,"\nyNum (w/ padding) ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout," %i ",ConstMem.YNum[i]);
   };
   fprintf(stdout,"\ndx ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout, " %.2f ",ConstMem.Dx[i]);
   };
   fprintf(stdout,"\ndy ");
   for (int i=0;i<MGS.RequestedLevels;i++) {
     fprintf(stdout," %.2f ",ConstMem.Dy[i]);
   };
   fprintf(stdout,"\nBoundaires ");
   for (int i=0;i<=3;i++) {
     fprintf(stdout," %i ",ConstMem.BC[i]);
   };
   fprintf(stdout,"\n");


   switch( MGS.equations) {
   case 0:
     fprintf(stdout,"Input file uses 2D Full Stokes Equations\n");
     #ifndef __2DSTOKES__
       std::cout << "Code is build for 2D Full Stokes equations but inputfile uses a different set of equations. This will not work! " << std::endl;
       exit(0);
     #endif
     break;

   case 1:
     fprintf(stdout,"Input file uses 3D Full Stokes Equations\n");
     fprintf(stdout, "\nEquations implemented! See Makefile for more infomation\n");
     exit(0);
     
     #ifndef __3DSTOKES__
     std::cout << "Code is build for 3D Full Stokes equations but inputfile uses a different set of equations. This wil not work! " << std::endl;
     exit(0);
     #endif

     break;

   case 2:
       fprintf(stdout,"Input file uses iSOSIA Equations\n");
       #ifndef __iSOSIA__
       std::cout << "Code is build for iSOSIA equations but inputfile uses a different set of equations. This will not work! " << std::endl;
       exit(0);
       #endif
     break;
   };
   
   fprintf(stdout, "\n////////////// RUNNING (%i,%i) threads per block on (%i,%i) grid  /////////////\n", MGS.threadsPerBlock[0], MGS.threadsPerBlock[1], MGS.numBlock[0], MGS.numBlock[1]);


   CurrentLevelNum = 0;
   if (MGS.RequestedLevels > 1) {
     Direction = 1;
   } else {
     Direction = 0;
   };
   
   // Set array to save residual in
   Residual = new ValueType[sizeof(ValueType)*MGS.NumberOfCycles];

   for (int i=0; i< MGS.NumberOfCycles; i++) {
     Residual[i] = 0.0;
   };

};

void Multigrid::InterpIceTopographyToCorner(const int level) {
  Levels[level]->InterpIceTopographyToCorner();
};

void Multigrid::SetConstMemOnDevice() 
{
  CustomError = 0; // Init error
  #ifdef __GPU__
    // Declare global memory space on device
    cudaMalloc( (void**) &d_ConstMem,  sizeof(ConstantMem) );
    checkForCudaErrors("Could not allocate constant memory structure in global memory!");
    
    cudaMalloc( (void**) &d_CustomError,  sizeof(int) ); // Allocate memory to handle custom error functions
    checkForCudaErrors("Could not allocate custom error variable in global memory!");
    
    cudaMemcpy( d_ConstMem   ,   &ConstMem,     sizeof(ConstantMem), cudaMemcpyHostToDevice );   // Copy structure
    checkForCudaErrors("Could not copy constant memory structure in global memory!");

    CustomError = 0; // Reset before copy

    cudaMemcpy( d_CustomError,   &CustomError,  sizeof(int),         cudaMemcpyHostToDevice );   // Copy error variable
    checkForCudaErrors("Could not copy custom error variable to global memory!");
    
    std::cout << "Tranfering Constant Memory to Device " << std::endl;
    //  std::cout << "Relaxs = " << ConstMem.Relaxs << std::endl;
    SetConstMem(&ConstMem);
    checkForCudaErrors("Creation of constants on device!", 1);
    
    std::cout << "MGS.GPUBlockSize " << MGS.GPUBlockSize.x << " " << MGS.GPUBlockSize.y << " " << MGS.GPUBlockSize.z << "\n";
    std::cout << "MGS.GPUGridSize " << MGS.GPUGridSize.x  << " " << MGS.GPUGridSize.y << " " <<  MGS.GPUGridSize.z  << "\n";

    cudaMemcpy( &CustomError,   d_CustomError,  sizeof(int),         cudaMemcpyDeviceToHost );   // Copy error variable back to host
    checkForCudaErrors("Could not copy custom error variable to host memory!", 1);
  #else
    SetConstMem(&ConstMem);
  #endif

  if (CustomError) {
    std::cout << "Custom error " << CustomError << "\n";
    Error(CustomError,"See SetConstMem or VerifyConstMem for infomation.");
  };
  std::cout << "Constant memory set on device" << std::endl;

};

MultigridStruct Multigrid::GetMultigridSettings() {
  return MGS;
};

/*! Add a level to the current Multigrid instance
 */
void Multigrid::AddLevels(const int maxStressIter) {


  // Create all requested levels
  std::cout << "Number of levels: " << MGS.RequestedLevels << std::endl;
  for (int i=0;i<MGS.RequestedLevels; i++) {
    if (CurrentLevelNum > maxLevel || CurrentLevelNum > MGS.RequestedLevels) {
      std::cout << "LOG: Cannot create more then " << maxLevel << " levels!" << std::endl;
      return;
    } else {
     
      Levels[CurrentLevelNum] = new LevelType(CurrentLevelNum,
					      ConstMem.XNum[CurrentLevelNum], ConstMem.YNum[CurrentLevelNum], 
					      ConstMem.Dx[CurrentLevelNum], ConstMem.Dy[CurrentLevelNum], 
					      maxStressIter,//MGS.IterationsOnLevel[CurrentLevelNum],
					      MGS.startIteration );
      
      Levels[CurrentLevelNum]->SetGPUBlockSize(MGS.threadsPerBlock[0],MGS.threadsPerBlock[1],1);
      Levels[CurrentLevelNum]->SetGPUGridSize(MGS.numBlock[0],MGS.numBlock[1],1);
      LastLevel = Levels[CurrentLevelNum];
      CurrentLevelNum++;
      
    std::cout << "LOG: New level created!" << std::endl;
    };
  };

  #ifdef __GPU__
    // Sync now
    cudaDeviceSynchronize();
  #endif

  ConstMem.LevelNum = CurrentLevelNum;
  checkForCudaErrors("Could not add level", 1);
};

void Multigrid::CopyLevelToDevice(const int CopyLevel) {
  Levels[CopyLevel]->CopyLevelToDevice();
};

void Multigrid::CopyLevelToHost(const int CopyLevel, const bool FileIO) {
   Levels[CopyLevel]->CopyLevelToHost(FileIO);
   std::cout << "Level Copied to Host!" << std::endl;
};

int MultigridLevel::GetLevel() {
  return LevelNum;
};

unsigned int Multigrid::GetCurrentLevels() {
  return CurrentLevelNum;
};

int Multigrid::GetNumberOfIterations() {
  return MGS.totIterNum;
};


void Multigrid::RestrictTopography(const int Fine, const int Coarse, const int CPU) {
    Levels[Coarse]->RestrictTopography(Levels[Coarse-1], Levels[Coarse]);

};

void MultigridLevel::SetGPUBlockSize(int x, int y, int z) {
  GPUBlockSize.x = x;
  GPUBlockSize.y = y;
  //  GPUGridSize.z = z;
};

void MultigridLevel::SetGPUGridSize(int x, int y, int z) {
  GPUGridSize.x = x;
  GPUGridSize.y = y;
  //  GPUGridSize.z = z;
};

void Multigrid::RunFMG(profile &prof) {

  std::cout << "Running Full Multigrid Cycle to get initial configuration \n";

  CurrentLevel                    = LastLevel;
  LevelType* CurrentLowestLevel   = LastLevel;
  LevelType* NextLevel            = NULL; //= Levels[CurrentLevel->LevelNum + Direction];
  LevelType* BaseLevel            = FirstLevel;

  // First restrict all levels with the right-hand side from first level
  // but this is zero, so just start from the last level
  
  // First relax on last level
  CurrentLevel->Iterate(1,1, prof, GetCycleNum(), MGS.Converg_s);

  // Go to level below
  NextLevel = Levels[CurrentLevel->LevelNum-1];
  CurrentLowestLevel = NextLevel;

  // Prolong result to level below
  std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
  prolongLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_Taue, 
	       NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
	       CurrentLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
	       CurrentLevel->LevelNum, 
	       NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
	       (CurrentLevel->XNum), (CurrentLevel->YNum), 1.0, 1.0, 0, MGS.ProlongOperator,
	       CurrentLevel, NextLevel
	       );

  CurrentLowestLevel = NextLevel;

  // Now run a V-cycle starting at each level until the first
  while(CurrentLowestLevel != BaseLevel) {    
    SetFirstLevel(CurrentLowestLevel->LevelNum);

    CurrentLowestLevel->Iterate(1,1, prof, GetCycleNum(), MGS.Converg_s);

    RunVCycle(prof,0,1);

    NextLevel = Levels[CurrentLowestLevel->LevelNum-1];

    // Prolong result to level below
    std::cout << "Running prolong from level " << CurrentLowestLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
    prolongLevel(CurrentLowestLevel->d_Vx, CurrentLowestLevel->d_Vy, CurrentLowestLevel->d_Taue, 
		 NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		 CurrentLowestLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
		 CurrentLowestLevel->LevelNum, 
		 NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLowestLevel->GPUGridSize, CurrentLowestLevel->GPUBlockSize, 
		 (CurrentLowestLevel->XNum), (CurrentLowestLevel->YNum), 1.0, 1.0, 0, MGS.ProlongOperator,
		 CurrentLowestLevel, NextLevel
		 );
    
    CurrentLowestLevel = NextLevel;
  };

  CurrentLowestLevel->Iterate(1,1, prof, GetCycleNum(), MGS.Converg_s);

};

void Multigrid::RunWCycle(profile & prof) {

  CurrentLevel = FirstLevel;
  LevelType* NextLevel = NULL;

  // First run to the lowest level
  while (CurrentLevel != LastLevel) {
    // Smooth current level
    CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
    
    NextLevel = Levels[CurrentLevel->LevelNum + Direction];

    #ifdef __iSOSIA__
    	  std::cout << "Running restrict from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA \n";	
	  
	  // Restrict error
	  restrictLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
			NextLevel->d_VxR, NextLevel->d_VyR, NextLevel->d_TaueR, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	  // Restrict physical value
	  restrictLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_Taue, 
			NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	 
	  NextLevel->ComputeStrainRate();
	  NextLevel->IterateStress(1, 0);
	  NextLevel->updateWantedOnLevel();
    #endif

	  CurrentLevel = NextLevel;
  };
  
  // Iterate last level
  //  CurrentLevel->Iterate(0, prof, GetCycleNum());

  std::cout << "Running first small prolong. There are " << LastLevel->LevelNum << " Levels" << std::endl;

  // Run first small V (bump)
  // to third lowest level and back
  // we need a total of 4 (0->3) level to do 4-cycle
  if (LastLevel->LevelNum > 2 ) {
    while (CurrentLevel != Levels[FirstLevel->LevelNum + 1] ) {
      NextLevel = Levels[CurrentLevel->LevelNum - 1];

      restrictLevel(NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		    CurrentLevel->d_VxR, CurrentLevel->d_VyR, CurrentLevel->d_TaueR, 
		    NextLevel->d_VxRes, CurrentLevel->d_VxRes, // Dummy values
		    CurrentLevel->LevelNum,
		    CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, (NextLevel->XNum)-2, (NextLevel->YNum)-2, 0, MGS.RestrictOperator,
		    NextLevel, CurrentLevel);
      
      // Compute error (e = u - v) and save to Vx, Vy or Taue
      CurrentLevel->ResetDeviceGrid(2); // Reset residual grids	  
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vx, CurrentLevel->d_VxR, CurrentLevel->d_VxRes, 1, 0);
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vy, CurrentLevel->d_VyR, CurrentLevel->d_VyRes, 0, 1);
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Taue, CurrentLevel->d_TaueR, CurrentLevel->d_TaueRes, 0, 0);
            
      //      std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
      prolongLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
		   NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		   CurrentLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
		   CurrentLevel->LevelNum, 
		   NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
		   (CurrentLevel->XNum), (CurrentLevel->YNum), ConstMem.vKoef, ConstMem.sKoef, 0, MGS.ProlongOperator,
		       CurrentLevel, NextLevel
		   );
      
      CurrentLevel->Iterate(0, 1,prof, GetCycleNum(), MGS.Converg_s);
      CurrentLevel = Levels[CurrentLevel->LevelNum -1];
    };
  };

  // Iterate level
  CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
  exit(0);
  // Send MG back to LastLevel
  while (CurrentLevel != LastLevel) {
    // Smooth current level
    CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
    NextLevel = Levels[CurrentLevel->LevelNum + 1];

    #ifdef __iSOSIA__
    //	  std::cout << "Running restrict from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA \n";	
	  
	  // Restrict error
	  restrictLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
			NextLevel->d_VxR, NextLevel->d_VyR, NextLevel->d_TaueR, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	  // Restrict physical value
	  restrictLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_Taue, 
			NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	 
	  NextLevel->ComputeStrainRate();
	  NextLevel->IterateStress(1, 0);
	  NextLevel->updateWantedOnLevel();
    #endif

	  CurrentLevel = NextLevel;
  };


  // Iterate last level
  CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
  exit(0);
  /* Run Large Bump */
  if (CurrentLevel->LevelNum > 1 ) {
    while (NextLevel != Levels[FirstLevel->LevelNum] ) {
	  
      restrictLevel(NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		    CurrentLevel->d_VxR, CurrentLevel->d_VyR, CurrentLevel->d_TaueR, 
		    NextLevel->d_VxRes, CurrentLevel->d_VxRes, // Dummy values
		    CurrentLevel->LevelNum,
		    CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, (NextLevel->XNum)-2, (NextLevel->YNum)-2, 0, MGS.RestrictOperator,
		    NextLevel, CurrentLevel);
      
      // Compute error (e = u - v) and save to Vx, Vy or Taue
      CurrentLevel->ResetDeviceGrid(2); // Reset residual grids	  
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vx, CurrentLevel->d_VxR, CurrentLevel->d_VxRes, 1, 0);
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vy, CurrentLevel->d_VyR, CurrentLevel->d_VyRes, 0, 1);
      CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Taue, CurrentLevel->d_TaueR, CurrentLevel->d_TaueRes, 0, 0);
      
      
      
      std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
      prolongLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
		   NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		   CurrentLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
		   CurrentLevel->LevelNum, 
		   NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
		   (CurrentLevel->XNum), (CurrentLevel->YNum), ConstMem.vKoef, ConstMem.sKoef, 0, MGS.ProlongOperator,
		       CurrentLevel, NextLevel
		   );
      
      CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
      NextLevel = Levels[CurrentLevel->LevelNum - 1];
    };

    CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
  };

  // Iterate second level
  CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
  exit(0);
  // Send MG back to LastLevel
  while (CurrentLevel != LastLevel) {
    // Smooth current level
    CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
    
    NextLevel = Levels[CurrentLevel->LevelNum + 1];

    #ifdef __iSOSIA__
    	  std::cout << "Running restrict from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA \n";	
	  
	  // Restrict error
	  restrictLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
			NextLevel->d_VxR, NextLevel->d_VyR, NextLevel->d_TaueR, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	  // Restrict physical value
	  restrictLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_Taue, 
			NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	 
	  NextLevel->ComputeStrainRate();
	  NextLevel->IterateStress(1, 0);
	  NextLevel->updateWantedOnLevel();
    #endif

	  CurrentLevel = NextLevel;
  };

  while (CurrentLevel != Levels[FirstLevel->LevelNum] ) {
	  
    restrictLevel(NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		  CurrentLevel->d_VxR, CurrentLevel->d_VyR, CurrentLevel->d_TaueR, 
		  NextLevel->d_VxRes, CurrentLevel->d_VxRes, // Dummy values
		  CurrentLevel->LevelNum,
		  CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, (NextLevel->XNum)-2, (NextLevel->YNum)-2, 0, MGS.RestrictOperator,
		  NextLevel, CurrentLevel);
    
    // Compute error (e = u - v) and save to Vx, Vy or Taue
    CurrentLevel->ResetDeviceGrid(2); // Reset residual grids	  
    CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vx, CurrentLevel->d_VxR, CurrentLevel->d_VxRes, 1, 0);
    CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vy, CurrentLevel->d_VyR, CurrentLevel->d_VyRes, 0, 1);
    CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Taue, CurrentLevel->d_TaueR, CurrentLevel->d_TaueRes, 0, 0);
    
    
    
    std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
    prolongLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
		 NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
		 CurrentLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
		 CurrentLevel->LevelNum, 
		 NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
		 (CurrentLevel->XNum), (CurrentLevel->YNum), ConstMem.vKoef, ConstMem.sKoef, 0, MGS.ProlongOperator,
		 CurrentLevel, NextLevel
		 );
    
    CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
    NextLevel = Levels[CurrentLevel->LevelNum - 1];
  };

  // Iterate final level when coming out of cycle
  CurrentLevel->Iterate(0,1, prof, GetCycleNum(), MGS.Converg_s);
  
};

int Multigrid::GSCycle(profile & prof, bool doMG,const int ForceVelocity) {

  CurrentLevel = FirstLevel;
  LevelType* NextLevel = Levels[CurrentLevel->LevelNum + Direction];

  CurrentLevel->glob_StressIterations = 0;

  // Run one last smoother on FirstLevel
  //  std::cout << "Last smoother \n";
  //  FirstLevel->SetBoundaries();    
  FirstLevel->Iterate(0, ForceVelocity, prof, 
		      MGS.IterationsOnLevel[CurrentLevel->LevelNum],
		      MGS.Converg_s);// GetCycleNum());

  // Do a sync before going into main function
  checkForCudaErrors("Post Multigrid Cycle", 1);

  return CurrentLevel->glob_StressIterations;
}

int Multigrid::RunVCycle(profile & prof, bool doMG, const int ForceVelocity) {

  CurrentLevel = FirstLevel;
  LevelType* NextLevel = Levels[CurrentLevel->LevelNum + Direction];

  CurrentLevel->glob_StressIterations = 0;
  
  while (NextLevel != FirstLevel) {
    //    std::cout << "\r Running V-Cycle on level " << CurrentLevel->LevelNum << std::endl;

    //    CurrentLevel->setResidual(0.0); // Reset residual variables
    //    if (CurrentLevel->LevelNum==0)
    CurrentLevel->SetBoundaries();    
    CurrentLevel->Iterate(0, ForceVelocity, prof, MGS.IterationsOnLevel[CurrentLevel->LevelNum],
			  MGS.Converg_s);//GetCycleNum());

    if ( (CurrentLevel == LastLevel && Direction > 0) || 
	 (CurrentLevel == FirstLevel && Direction < 0) )
      Direction *= -1;

    if ( (FirstLevel->LevelNum <= CurrentLevel->LevelNum + Direction) && 
	 (CurrentLevel->LevelNum + Direction <= LastLevel->LevelNum) &&
	 (CurrentLevelNum != 1) &&
	 doMG
	 ) {

      NextLevel = Levels[CurrentLevel->LevelNum + Direction];
      
	// Restrict or Prolong	
	if (Direction > 0 && NextLevel != FirstLevel) {
	  
	  // Set grid variables to zero
	  NextLevel->ResetDeviceGrid();
	  
	  // Coarsen level 0 -> 1
	  // CurrentLevel->Restrict();
	  //  std::cout << "\n";
	  
#ifdef __2DSTOKES__
	  //	  std::cout << "Running restrict from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using 2D Stokes \n";	
	  restrictLevel(CurrentLevel->d_Resx, CurrentLevel->d_Resy, CurrentLevel->d_Resc, 
			NextLevel->d_Rx, NextLevel->d_Ry, NextLevel->d_Rc, 
			CurrentLevel->d_Etan, NextLevel->d_Etan, NextLevel->LevelNum, 
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
			CurrentLevel, NextLevel);
#elif defined __iSOSIA__ 		       
	  //  std::cout << "Running restrict from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA \n";	

	  //	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_TaueRes, "taueRes_prerestrict", CurrentLevel->LevelNum);	  	 	  	  

	  NextLevel->ResetDeviceGrid(1);	  

	  // Restrict error
	  restrictLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
			NextLevel->d_VxR, NextLevel->d_VyR, NextLevel->d_TaueR, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);

	  // Restrict physical value
	  restrictLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_Taue, 
			NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
			CurrentLevel->d_VxRes, NextLevel->d_VxRes, // Dummy values
			NextLevel->LevelNum,
			NextLevel->GPUGridSize, NextLevel->GPUBlockSize, (CurrentLevel->XNum)-2, (CurrentLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		CurrentLevel, NextLevel);
	  /*
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_TaueRes, "taueRes_restrict", NextLevel->LevelNum);	  	 
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_TaueR, "taueR_restrict", NextLevel->LevelNum);	  	 
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Taue, "taue_restrict", NextLevel->LevelNum);	  	 
	  */

	  // Force new stress field to converge
	  // before computing the new right hand side
	  /*
	  NextLevel->Iterate(1, prof, 
			     MGS.IterationsOnLevel[CurrentLevel->LevelNum],
			     MGS.Converg_s);
	  */
	  /*
	  NextLevel->ComputeStrainRate();
	  NextLevel->IterateStress(1, 0);
	  */
	  CurrentLevel->SetBoundaries();    
	  NextLevel->updateWantedOnLevel();

	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Vx, "Vx_postUpdate", NextLevel->LevelNum);	  	 
	  NextLevel->cloneArray(NextLevel->d_Vx, NextLevel->d_Vxc); // Copy the array for prolong
	  NextLevel->cloneArray(NextLevel->d_Vy, NextLevel->d_Vyc); // Copy the array for prolong

#endif

	} else if ( NextLevel != CurrentLevel ) { 

	  // CurrentLevel->Prolong();
#ifdef __2DSTOKES__
	  //  std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using 2D Stokes setup \n";	
	  prolongLevel(CurrentLevel->d_Vx, CurrentLevel->d_Vy, CurrentLevel->d_P, 
		       NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_P, 
		       CurrentLevel->d_Etan, NextLevel->d_Etan, CurrentLevel->LevelNum, 
		       NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
		       (CurrentLevel->XNum), (CurrentLevel->YNum), ConstMem.vKoef, ConstMem.pKoef, 0, MGS.ProlongOperator,
		       CurrentLevel, NextLevel);
#elif defined __iSOSIA__
	  //	  std::cout << "Running restrict before prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA \n";	
	  // Restrict physical value to coarse grid again to compute error
	  /*
	  restrictLevel(NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue, 
			CurrentLevel->d_VxR, CurrentLevel->d_VyR, CurrentLevel->d_TaueR, 
			NextLevel->d_VxRes, CurrentLevel->d_VxRes, // Dummy values
			CurrentLevel->LevelNum,
			CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, (NextLevel->XNum)-2, (NextLevel->YNum)-2, 0, MGS.RestrictOperator,
	  		NextLevel, CurrentLevel);
	  */
	  // Set BC in the new d_VxR and d_VyR vectors
	  //	  CurrentLevel->SetBoundaries(5);

	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_Vxc, "vxc_Prolong", CurrentLevel->LevelNum);	  
	  // Compute error (e = u - v) and save to Vx, Vy or Taue
	  CurrentLevel->ResetDeviceGrid(2); // Reset residual grids	  
	  CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vx, CurrentLevel->d_Vxc, CurrentLevel->d_VxRes, 1, 0);
	  CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Vy, CurrentLevel->d_Vyc, CurrentLevel->d_VyRes, 0, 1);
	  CurrentLevel->ComputeErrorForProlong(CurrentLevel->d_Taue, CurrentLevel->d_TaueR, CurrentLevel->d_TaueRes, 0, 0);	  
	  
	  //CurrentLevel->SetBoundaries(5);
	  //	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_Taue, "taue_Prolong", CurrentLevel->LevelNum);	  
	  //	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_TaueR, "taueR_Prolong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VxRes, "vxRes_Prolong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VxR, "vxR_Prolong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VyR, "vyR_Prolong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_Vx, "vx_Prolong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VyRes, "vyRes_Prolong", CurrentLevel->LevelNum);	  
	  //	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_TaueRes, "taueRes_Prolong", CurrentLevel->LevelNum);	
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Vx, "vx_preProlong", NextLevel->LevelNum);	
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Vy, "vy_preProlong", NextLevel->LevelNum);	
  
	  // Try to smooth vyRes before prolong
	  //CurrentLevel->FixVelocityGradients(3);

	  //	  std::cout << "Running prolong from level " << CurrentLevel->LevelNum << " -> " << NextLevel->LevelNum << " using iSOSIA setup \n";	
	  prolongLevel(CurrentLevel->d_VxRes, CurrentLevel->d_VyRes, CurrentLevel->d_TaueRes, 
		       NextLevel->d_Vx, NextLevel->d_Vy, NextLevel->d_Taue,  
		       CurrentLevel->d_Vx, NextLevel->d_Vx,  // Dummy values
		       CurrentLevel->LevelNum, 
		       NextLevel->GPUGridSize, NextLevel->GPUBlockSize, CurrentLevel->GPUGridSize, CurrentLevel->GPUBlockSize, 
		       (CurrentLevel->XNum), (CurrentLevel->YNum), ConstMem.vKoef, ConstMem.sKoef, 0, MGS.ProlongOperator,
		       CurrentLevel, NextLevel
		       );
	  
	  // Run a single iteration of velocity
	  // to smooth interpolation error
	  /*
	  NextLevel->Iterate(0, prof, 1,
				1e30);
	  */
	  //	  NextLevel->SetBoundaries();


	  // Smooth the edges of the velocity field
	  /*
	  NextLevel->FixVelocityGradients(1);
	  NextLevel->FixVelocityGradients(2);
	  */

	  NextLevel->SetBoundaries(0);
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Vx, "vx_postProlong", NextLevel->LevelNum);	  
	  NextLevel->dumpGPUArrayToFile(NextLevel->d_Vy, "vy_postProlong", NextLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VxRes, "VxRes_postProlong", CurrentLevel->LevelNum);	  
	  CurrentLevel->dumpGPUArrayToFile(CurrentLevel->d_VyRes, "VyRes_postProlong", CurrentLevel->LevelNum);	  
	  //	  exit(0);
	  
#endif
	};
	
	CurrentLevel = NextLevel;
    } else {
      // Special case with only one grid
      NextLevel = FirstLevel;
      //return;
    };
  };


  // Run one last smoother on FirstLevel
  //  std::cout << "Last smoother \n";
  //  FirstLevel->SetBoundaries();    
  FirstLevel->Iterate(0,ForceVelocity,prof, 
		      MGS.IterationsOnLevel[CurrentLevel->LevelNum],
		      MGS.Converg_s);// GetCycleNum());

  if (CurrentLevelNum == 1 && 0) {
    FirstLevel->Iterate(0,ForceVelocity, prof, 
			MGS.IterationsOnLevel[CurrentLevel->LevelNum],
			MGS.Converg_s);// GetCycleNum());
  };

  // Do a sync before going into main function
  checkForCudaErrors("Post Multigrid Cycle", 1);

  return CurrentLevel->glob_StressIterations;
};


void Multigrid::SolveHydrology(double dt) {  
  // Save old value
  Levels[MGS.startLevel]->cloneArray(Levels[MGS.startLevel]->d_hw,
				     Levels[MGS.startLevel]->d_hw_old);
  Levels[MGS.startLevel]->cloneArray(Levels[MGS.startLevel]->d_Pw,
				     Levels[MGS.startLevel]->d_Pw_old);
  
  Levels[MGS.startLevel]->solveHydrology(dt);
  //  res = Levels[MGS.startLevel]->HydrologyResidual_h();

};

ValueType Multigrid::VelocityResidual(profile &prof) {
  return Levels[MGS.startLevel]->VelocityResidual(prof);
  
};

ValueType Multigrid::SlidingResidual() {
  return Levels[MGS.startLevel]->SlidingResidual();
  
};

void Multigrid::ResidualCheck(const int CurrentIter,            // Current Iteration
			      const ValueType CurrentResidual,  // Residual of current iteration
			      int &NextIter,                    // Next time to update residual
			      int &LastIter,                    // Last time residual was updated
			      ValueType &LastResidual,          // Residual at last update
			      int &ExpectedConvergence,         // The cycle where convergence is expected
			      int StartIteration,               // Number of iterations to run before computing first residual
			      double StpLen,                    // Lenght of step before next update
			      int SingleStp,                    // When to compute single steps
			      double ConvergenceRule            // When has convergence happend (normally 10^-3)
			      ) {
  // Forward call to iSOSIA class
  Levels[MGS.startLevel]->ResidualCheck(CurrentIter,
					CurrentResidual,
					NextIter,
					LastIter,
					LastResidual,
					ExpectedConvergence,
					MGS.FirstStp,
					MGS.StpLen,
					MGS.SingleStp,
					ConvergenceRule
					);
};

void Multigrid::ComputeSurfaceVelocity(const int updateType) {
  Levels[0]->ComputeSurfaceVelocity(updateType);
};

int Multigrid::NumberOfCycles() {
  return MGS.NumberOfCycles;
};

void Multigrid::SetFirstLevel(int Level) {
  FirstLevel = Levels[Level];
};

void Multigrid::SetFirstLevel( ) {
  FirstLevel = Levels[MGS.startLevel];
};

int Multigrid::GetFirstLevel( ) {
  return MGS.startLevel;
};



int Multigrid::GetIOUpdate() {
  return MGS.IOUpdate;
};

void MultigridLevel::SetDimensions(int Xnum, int Ynum) {
  XNum = Xnum;
  YNum = Ynum;
};


void Multigrid::ComputeMaxError(int level) {
  
  ValueType XError, YError, PError;
  Levels[level]->MaxError(XError, YError, PError);

  std::cout << "Max Residual on level " << level << std::endl;
  std::cout << "X Max = " << XError << " Y Max = " << YError << " P = " << PError << std::endl;
};


ValueType Multigrid::ComputeNorm(int level) {
  
  ValueType XNorm = 0.0;
  ValueType YNorm = 0.0;
  ValueType PNorm = 0.0;
  //  Levels[level]->L2Norm(XNorm, YNorm, PNorm);

  std::cout << "L2 Norm on level " << level << std::endl;
  std::cout << "||X|| = " << XNorm << " ||Y|| = " << YNorm << " ||P|| = " << PNorm << std::endl;
  return XNorm;
};

//
// Common MultigridLevel Functions
//

void MultigridLevel::Memset(ValueType * Grid, double Val) {
  for (int i=0;i<XNum;i++) {
    for (int j=0;j<YNum;j++) {
      (Grid[i*XNum+j]) = Val;
    };
  };
};

void Multigrid::SetCycleNum(const int num) {
  if (num < 0) {
    MGS.Cycle = MGS.startIteration;
  } else {
    MGS.Cycle = num;
  };
};

void Multigrid::ReduceRelaxation(const ValueType factor) {
  ConstMem.Relaxs *= factor;
  ConstMem.Relaxv *= factor;
  
  if (ConstMem.Relaxs < 1e-6 || ConstMem.Relaxv < 1e-6) {
    std::cout << "Values for Relaxs and/or Relaxv are now below 1e-6. This will not converge!" << "\n" << "I am quitting. \n";
    exit(0);
  };

  SetConstMemOnDevice();
  std::cout << "Residual is growing reducing relaxation to try and help" << "\n";
  std::cout << "New values are: Relaxs " << ConstMem.Relaxs << " Relaxv " << ConstMem.Relaxv << "\n";
};

void Multigrid::setResidual(const ValueType Res, const int CycleNum) {
  Residual[CycleNum] = Res;
};

ValueType Multigrid::getResidual(const int CycleNum) {
  return Residual[CycleNum];
};

void Multigrid::UpdateCycleNum() {
  MGS.Cycle++;
};

int Multigrid::GetCycleNum() {
  return MGS.Cycle;
};

void MultigridLevel::Print(ValueType * Grid) {
  for (int i=0;i<XNum;i++) {
    for (int j=0;j<YNum;j++) {
      printf("%e ",Grid[i*XNum+j]);
    }
    printf("\n");
  }
};

MultigridLevel::MultigridLevel(const unsigned int Num, 
			       const unsigned int xNum, const unsigned int yNum, 
			       const ValueType Dx, const ValueType Dy, 
			       const unsigned int IterationLevel,
			       const unsigned int startIteration)
: LevelNum(Num), XNum(xNum), YNum(yNum), dx(Dx), dy(Dy), SmootherIterations(IterationLevel)
{

  // TODO: Check LevelNum to be lower then MaxLevels
  GridSize = sizeof(ValueType)*( XNum*YNum );

  SmootherIterations = IterationLevel;

  if (GridSize <= 0) {
    Error(-6,"Dimensions of grid is zero or negative");
    exit(0);
  };
};




/*! Default constructor
 *
 */
MultigridLevel::~MultigridLevel() {

};

void Multigrid::restrictLevel(ValueType *d_Resx, ValueType *d_Resy, ValueType *d_Resc, 
			      ValueType *d_Rx, ValueType *d_Ry, ValueType *d_Rc, 
			      ValueType *d_Etanf, ValueType *d_Etanc, 
			      int level, dim3 grid_size, dim3 block_size, 
			      int XNum, int YNum, int verbose, int RestrictOperator,
			      LevelType *CurrentLevel, LevelType *NextLevel) {

  
  // Fill empty nodes of fine level 
  /*
  CurrentLevel->FillGhost(d_Rx, 2, 1); // Pointer, XStart, YStart
  CurrentLevel->FillGhost(d_Ry, 1, 2); // Pointer, XStart, YStart
  CurrentLevel->FillGhost(d_Rc, 1, 1); // Pointer, XStart, YStart
*/
  switch (RestrictOperator) {
  case 0:
    #ifdef __iSOSIA__
    std::cout << "iSOSIA not tested on this restriction \n";
    exit(0);
    #endif
    /* Half weighted */
    std::cout << "Running half weight restrict from level " << level-1 << " -> " << level << std::endl;
    if (verbose) msg(stdout,"Restricting Resx to Rx",1,0);
    Restrict DEVICESPECS (d_Resx, d_Rx, level, 0, 1, RestrictOperator);
    checkForCudaErrors("Restriction of Resx",0);
    if (verbose) msg(stdout,"Restricting Resy to Ry",1,0);
    Restrict DEVICESPECS (d_Resy, d_Ry,level, 1, 0, RestrictOperator);
    checkForCudaErrors("Restriction of Resy",0);
    if (verbose) msg(stdout,"Restricting Resc to Rc",1,0);
    Restrict DEVICESPECS (d_Resc, d_Rc,level, 0, 0, RestrictOperator);
    checkForCudaErrors("Restriction of Resc",0);
    break;
    
  case 1:
    #ifdef __iSOSIA__
    std::cout << "iSOSIA not tested on this restriction \n";
    exit(0);
    #endif

    /* Full weighted */
    std::cout << "Running full weight restrict from level " << level-1 << " -> " << level << std::endl;
    if (verbose) msg(stdout,"Restricting Resx to Rx",1,0);
    Restrict DEVICESPECS (d_Resx, d_Rx, level, 0, 1, RestrictOperator);
    checkForCudaErrors("Restriction of Resx",0);
    if (verbose) msg(stdout,"Restricting Resy to Ry",1,0);
    Restrict DEVICESPECS (d_Resy, d_Ry,level, 1, 0, RestrictOperator);
    checkForCudaErrors("Restriction of Resy",0);
    if (verbose) msg(stdout,"Restricting Resc to Rc",1,0);
    Restrict DEVICESPECS (d_Resc, d_Rc,level, 0, 0, RestrictOperator);
    checkForCudaErrors("Restriction of Resc",0);
    break;

  case 2:
    /* Bilinear */
    #ifdef __iSOSIA__
    //    std::cout << "Running bilinear restrict from level " << level-1 << " -> " << level << std::endl;
    if (verbose) msg(stdout,"Restricting ResVx to RVx",1,0);
    restriction DEVICESPECS (d_Resx, d_Rx, 0.5, 0.0, 0, d_Resx, d_Rx, level, 1, 0);
    checkForCudaErrors("Restriction of ResVx",0);
    if (verbose) msg(stdout,"Restricting ResVy to RVy",1,0);
    restriction DEVICESPECS  (d_Resy, d_Ry, 0.0, 0.5, 0, d_Resy, d_Ry, level, 0, 1);
    checkForCudaErrors("Restriction of ResVy",0);
    if (verbose) msg(stdout,"Restricting Resc to TaueR",1,0);
    restriction DEVICESPECS (d_Resc, d_Rc, 0.5, 0.5, 0, // Use Eta? - NO
			     d_Resc, d_Rc, level, 
			     0, 0); // LB & UB
    
    #elif defined __2DSTOKES__
    //    std::cout << "Running bilinear restrict from level " << level-1 << " -> " << level << std::endl;
    if (verbose) msg(stdout,"Restricting Resx to Rx",1,0);
    restriction DEVICESPECS (d_Resx, d_Rx, 0.0, 0.5, 0, d_Etanf, d_Etanc, level, 1, 0);
    checkForCudaErrors("Restriction of Resx",0);
    if (verbose) msg(stdout,"Restricting Resy to Ry",1,0);
    restriction DEVICESPECS  (d_Resy, d_Ry, 0.5, 0.0, 0, d_Etanf, d_Etanc, level, 0, 1);
    checkForCudaErrors("Restriction of Resy",0);
    if (verbose) msg(stdout,"Restricting Resc to Rc",1,0);
    restriction DEVICESPECS (d_Resc, d_Rc, 0.5, 0.5, 1, // Use Eta? - YES
			     d_Etanf, d_Etanc, level, 
			     0, 0); // LB & UB
    #endif
    checkForCudaErrors("Restriction of Resc",0);
    break;
  default:
    std::cout << "Restrict operator must be either 0, 1 or 2!" << std::endl;
    return;
  };

  // Fill empty nodes of fine level 

  /*
  NextLevel->FillGhost(d_Rx, 1, 2); // Pointer, XStart, YStart
  NextLevel->FillGhost(d_Ry, 2, 1); // Pointer, XStart, YStart
  NextLevel->FillGhost(d_Rc, 1, 1); // Pointer, XStart, YStart
  */

  #ifdef __GPU__
  cudaDeviceSynchronize();
  #endif

};

void Multigrid::prolongLevel(ValueType *d_Vxc, ValueType *d_Vyc, ValueType *d_Pc, 
		  ValueType *d_Vxf, ValueType *d_Vyf, ValueType *d_Pf, 
		  ValueType *d_Etanc, ValueType *d_Etanf, 
		  int level, 
		  dim3 grid_sizef, dim3 block_sizef, dim3 grid_sizec, dim3 block_sizec, 
		  int XNum, int YNum, 
		  ValueType vkoef, ValueType pkoef, 
		  int verbose, int ProlongOperator,
		  LevelType *CurrentLevel, LevelType *NextLevel) {

  if (verbose)
    msg(stdout,"Prolonging Vx to Vx",1,0);

  // NOTE: Level is coarse

  // 0 - Tranfere overlapping nodes
  // 1 - Do vertical and horisontal interpolation
  // 2 - Interpolate rest
  //   for (int mode = 0; mode < 3; mode++) {
  int mode = 1;
  
  // Fill empty nodes of coarse level   
  
  CurrentLevel->FillGhost(d_Vxc, 1, 2, 0); // Pointer, XStart, YStart
  CurrentLevel->FillGhost(d_Vyc, 2, 1, 0); // Pointer, XStart, YStart
  CurrentLevel->FillGhost(d_Pc, 1, 1, 0); // Pointer, XStart, YStart
  
  checkForCudaErrors("Filling ghost nodes before prolong", 0);
    
  /*
  std::cout << "Prolong grid_sizef " << grid_sizef.x << " blocksize " << block_sizef.x << "\n";
  std::cout << "Using vKoef " << vkoef << " and pKoef " << pkoef << "\n"; 
  */

  prolongation
#ifdef __GPU__
    <<<grid_sizef, block_sizef>>>
#endif
    (d_Vxf, d_Vxc, 0.0, 0.5, 0, d_Etanf, d_Etanc, vkoef, level, 1, 0, mode);
  checkForCudaErrors("Prolongation of Vx", 0);
  
  if (verbose)
    msg(stdout,"Prolonging Vy to Vy",1,0);
  
  
  prolongation 
#ifdef __GPU__
    <<<grid_sizef, block_sizef>>>
#endif
    (d_Vyf, d_Vyc, 0.5, 0.0, 0, d_Etanf, d_Etanc, vkoef, level, 0, 1, mode);
  checkForCudaErrors("Prolongation of Vy", 0);
  
  if (verbose)
    msg(stdout,"Prolonging P to P",1,0);
  
  prolongation 
#ifdef __GPU__
    <<<grid_sizef, block_sizef>>>
#endif
    (d_Pf, d_Pc, 0.5, 0.5, 0, d_Etanf, d_Etanc, pkoef, level, 0, 0, mode);
  checkForCudaErrors("Prolongation of P", 0);
  
  #ifdef __GPU__
  cudaDeviceSynchronize();
  #endif

};
