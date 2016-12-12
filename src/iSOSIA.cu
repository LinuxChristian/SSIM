/* iSOSIA Equations */
 
#ifdef __iSOSIA__

#ifdef __GPU__
//#include <cuda_profiler_api.h>
#endif

#include "config.h"
#include <math.h>

/*! \brief Constructor for iSOSIA level subclass
 *
 */

iSOSIA::iSOSIA(const unsigned int Num,
	       const unsigned int xNum, const unsigned int yNum, 
	       const ValueType Dx, const ValueType Dy, 
	       const unsigned int IterationLevel,
	       const unsigned int startIteration)
: MultigridLevel(Num, xNum, yNum, Dx, Dy, IterationLevel, startIteration)
{

  checkForCudaErrors("Just about to allocate memory", 1);
  if (GridSize <= 0) {
    Error(-6,"Dimensions of grid is zero or negative");
    exit(0);
  };
  std::cout << "GridSize = " << GridSize/pow(1000,2) << " MB" << std::endl;

  glob_StressIterations = 0;
  
  // Allocate 2D grid on CPU
  IceTopo = new ValueType[GridSize];
  IceTopom= new ValueType[GridSize];
  BedTopo = new ValueType[GridSize];   
  Sxx     = new ValueType[GridSize];
  Syy     = new ValueType[GridSize];
  Sxy     = new ValueType[GridSize];


  // Mean interpolated values
  Sxxm    = new ValueType[GridSize];
  Syym    = new ValueType[GridSize];
  Sxym    = new ValueType[GridSize];

  // Velocity
  Vx      = new ValueType[GridSize];
  Vy      = new ValueType[GridSize];

  Vxd     = new ValueType[GridSize];
  Vyd     = new ValueType[GridSize];

  // Velocity from coarse grid (Used for prolong)
  Vxc     = new ValueType[GridSize];
  Vyc     = new ValueType[GridSize];

  // Surface velocity
  Vxs     = new ValueType[GridSize];
  Vys     = new ValueType[GridSize];

  // Basal/sliding velocity
  Vxb          = new ValueType[GridSize];
  VxbSuggested = new ValueType[GridSize];
  VxbR         = new ValueType[GridSize];
  VxbRes       = new ValueType[GridSize];

  Vyb          = new ValueType[GridSize];
  VybSuggested = new ValueType[GridSize];
  VybR         = new ValueType[GridSize];
  VybRes       = new ValueType[GridSize];

  VxRes   = new ValueType[GridSize];
  VyRes   = new ValueType[GridSize];
  VxSuggested = new ValueType[GridSize];
  VySuggested = new ValueType[GridSize];
  VxR     = new ValueType[GridSize];
  VyR     = new ValueType[GridSize];

  Exx     = new ValueType[GridSize];
  Eyy     = new ValueType[GridSize];
  Exy     = new ValueType[GridSize];

  Exyc    = new ValueType[GridSize]; // Exy in the corner (base) node
  Ezz     = new ValueType[GridSize];
  
  Taue    = new ValueType[GridSize];
  TaueRes = new ValueType[GridSize];
  TaueR   = new ValueType[GridSize];
  TaueSuggested = new ValueType[GridSize];

  // Hydrology variables
  Scx            = new ValueType[GridSize];
  ScxRes         = new ValueType[GridSize];
  ScxR           = new ValueType[GridSize];
  ScxSuggested   = new ValueType[GridSize];

  Scy            = new ValueType[GridSize];
  ScyR           = new ValueType[GridSize];
  ScyRes         = new ValueType[GridSize];
  ScySuggested   = new ValueType[GridSize];

  hw            = new ValueType[GridSize];
  hw_old        = new ValueType[GridSize];
  hwRes         = new ValueType[GridSize];
  hwR           = new ValueType[GridSize];
  hwSuggested   = new ValueType[GridSize];

  Pw            = new ValueType[GridSize];
  Pw_old        = new ValueType[GridSize];
  PwRes         = new ValueType[GridSize];
  PwR           = new ValueType[GridSize];
  PwSuggested   = new ValueType[GridSize];

  qsx   = new ValueType[GridSize];
  qsy   = new ValueType[GridSize];
  Qcx   = new ValueType[GridSize];
  Qcy   = new ValueType[GridSize];
  R   = new ValueType[GridSize];
  Psi   = new ValueType[GridSize];
  Mcx   = new ValueType[GridSize];
  Mcy   = new ValueType[GridSize];

  for (int i = 0;i<XNum;i++) {
    for (int j = 0;j<YNum;j++) {
      Sxx[j*XNum+i] = 0.0;
      Syy[j*XNum+i] = 0.0;
      Sxy[j*XNum+i] = 0.0;
      Vx[j*XNum+i] = 0.0;
      Vy[j*XNum+i] = 0.0;
      Vxd[j*XNum+i] = 0.0;
      Vyd[j*XNum+i] = 0.0;
      Vxs[j*XNum+i] = 0.0;
      Vys[j*XNum+i] = 0.0;
      Vxb[j*XNum+i] = 0.0;
      Vyb[j*XNum+i] = 0.0;
      Vxc[j*XNum+i] = 0.0;
      VxbSuggested[j*XNum+i] = 0.0;
      VxbR[j*XNum+i]         = 0.0;
      VxbRes[j*XNum+i]       = 0.0;
      VybSuggested[j*XNum+i] = 0.0;
      VybR[j*XNum+i]         = 0.0;
      VybRes[j*XNum+i]       = 0.0;


      IceTopo[j*XNum+i] = 0.0;
      IceTopom[j*XNum+i] = 0.0;
      BedTopo[j*XNum+i] = 0.0;

      Vyc[j*XNum+i] = 0.0;
      VxRes[j*XNum+i] = 0.0;
      VyRes[j*XNum+i] = 0.0;
      VxSuggested[j*XNum+i] = 0.0;
      VySuggested[j*XNum+i] = 0.0;
      VxR[j*XNum+i] = 0.0;
      VyR[j*XNum+i] = 0.0;

      Exx[j*XNum+i] = 0.0;
      Eyy[j*XNum+i] = 0.0;
      Exy[j*XNum+i] = 0.0;
      Exyc[j*XNum+i] = 0.0;
      Ezz[j*XNum+i] = 0.0;

      Taue[j*XNum+i] = 1.0;
      TaueR[j*XNum+i] = 0.0;
      TaueRes[j*XNum+i] = 0.0;
      TaueSuggested[j*XNum+i] = 0.0;

      // Hydrology variables
      Scx[j*XNum+i] = 0.0;
      ScxR[j*XNum+i] = 0.0;
      ScxRes[j*XNum+i] = 0.0;
      ScxSuggested[j*XNum+i] = 0.0;

      Scy[j*XNum+i] = 0.0;
      ScyRes[j*XNum+i] = 0.0;
      ScyR[j*XNum+i] = 0.0;
      ScySuggested[j*XNum+i] = 0.0;

      hw[j*XNum+i] = 1e-3;  // meter
      hw_old[j*XNum+i] = 0.0;  // meter
      hwRes[j*XNum+i] = 0.0; 
      hwR[j*XNum+i] = 0.0; 
      hwSuggested[j*XNum+i] = 0.0; 

      Pw[j*XNum+i] = 0.0; 
      Pw_old[j*XNum+i] = 0.0;  // meter
      PwRes[j*XNum+i] = 0.0; 
      PwR[j*XNum+i] = 0.0; 
      PwSuggested[j*XNum+i] = 0.0; 

      qsx[j*XNum+i] = 0.0;
      qsy[j*XNum+i] = 0.0;
      Qcx[j*XNum+i] = 0.0;
      Qcy[j*XNum+i] = 0.0;
      R[j*XNum+i] = 0.0;
      Psi[j*XNum+i] = 0.0;
      Mcx[j*XNum+i] = 0.0;
      Mcy[j*XNum+i] = 0.0;
    }
  }

  // Gradients

  dhdx = new ValueType[GridSize];
  dbdx = new ValueType[GridSize];
  dhdy = new ValueType[GridSize];
  dbdy = new ValueType[GridSize];

  dhdx_c = new ValueType[GridSize];
  dbdx_c = new ValueType[GridSize];
  dhdy_c = new ValueType[GridSize];
  dbdy_c = new ValueType[GridSize];  

  // Sliding is only done with SPM
  // variables are therefore declared in
  // spm.c and pointers are fixed.
  Tbx = NULL;
  Tby = NULL;
  Tbz = NULL;
  Ts  = NULL;
  Tn  = NULL;
  Vb  = NULL;
  //Pw  = NULL;

  ResidualVec = new ValueType[YNum*sizeof(ValueType)];
  MaxResVec = new ValueType[YNum*sizeof(ValueType)];
  
  // Setup residual variables
  ResTaueNorm = 1e10;
  ResVxNorm = 1e10;
  ResVyNorm = 1e10;

  // Setup residual check variables
  StressNextResidualUpdate = 1;
  StressLastResidualUpdate = -1;
  StressLastResidual = 1;
  StressExpectedConvergence = 0;

  #ifdef __GPU__
  checkForCudaErrors("Checking for errors before allocating GPU memory!");
  // Allocate Device memory only in CPU Mode
    cudaMalloc( (void**) &d_IceTopo,  GridSize );
    cudaMalloc( (void**) &d_IceTopom, GridSize );
    cudaMalloc( (void**) &d_BedTopo,  GridSize );
    cudaMalloc( (void**) &d_Sxx,      GridSize );
    cudaMalloc( (void**) &d_Syy,      GridSize );
    cudaMalloc( (void**) &d_Sxy,      GridSize );

    // Mean interpolated values
    cudaMalloc( (void**) &d_Sxxm,     GridSize );
    cudaMalloc( (void**) &d_Syym,     GridSize );
    cudaMalloc( (void**) &d_Sxym,     GridSize );

    // Default set to 1^10
    cudaMalloc( (void**) &d_ResTaueNorm, sizeof(ValueType) );
    cudaMalloc( (void**) &d_ResVxNorm,   sizeof(ValueType) );
    cudaMalloc( (void**) &d_ResVyNorm,   sizeof(ValueType) );

    // Surface velocity
    cudaMalloc( (void**) &d_Vxs,       GridSize );
    cudaMalloc( (void**) &d_Vys,       GridSize );

    // Values from coarse grids
    cudaMalloc( (void**) &d_Vxc,       GridSize );
    cudaMalloc( (void**) &d_Vyc,       GridSize );

    // Basal velocity
    cudaMalloc( (void**) &d_Vxb,          GridSize );
    cudaMalloc( (void**) &d_VxbSuggested, GridSize );
    cudaMalloc( (void**) &d_VxbRes,       GridSize );
    cudaMalloc( (void**) &d_VxbR,         GridSize );
    cudaMalloc( (void**) &d_Vyb,       GridSize );
    cudaMalloc( (void**) &d_VybSuggested, GridSize );
    cudaMalloc( (void**) &d_VybRes,       GridSize );
    cudaMalloc( (void**) &d_VybR,         GridSize );

    cudaMalloc( (void**) &d_Vxd,       GridSize );
    cudaMalloc( (void**) &d_Vyd,       GridSize );

    cudaMalloc( (void**) &d_Vx,       GridSize );
    cudaMalloc( (void**) &d_Vy,       GridSize );
    cudaMalloc( (void**) &d_VxRes,    GridSize );
    cudaMalloc( (void**) &d_VyRes,    GridSize );
    cudaMalloc( (void**) &d_VySuggested,      GridSize );
    cudaMalloc( (void**) &d_VxSuggested,      GridSize );    
    cudaMalloc( (void**) &d_Exx,      GridSize );
    cudaMalloc( (void**) &d_Eyy,      GridSize ); 
    cudaMalloc( (void**) &d_Exy,      GridSize ); 
    cudaMalloc( (void**) &d_Exyc,     GridSize ); 
    cudaMalloc( (void**) &d_Ezz,      GridSize ); 
    cudaMalloc( (void**) &d_Taue,     GridSize ); 
    cudaMalloc( (void**) &d_TaueRes,  GridSize );
    cudaMalloc( (void**) &d_TaueSuggested,  GridSize );
    cudaMalloc( (void**) &d_VyR,      GridSize );
    cudaMalloc( (void**) &d_VxR,      GridSize );
    cudaMalloc( (void**) &d_TaueR,    GridSize );

    /* Gradients */
    cudaMalloc( (void**) &d_dhdx,    GridSize );
    cudaMalloc( (void**) &d_dbdx,    GridSize );
    cudaMalloc( (void**) &d_dhdy,    GridSize );
    cudaMalloc( (void**) &d_dbdy,    GridSize );
    cudaMalloc( (void**) &d_dhdx_c,    GridSize );
    cudaMalloc( (void**) &d_dbdx_c,    GridSize );
    cudaMalloc( (void**) &d_dhdy_c,    GridSize );
    cudaMalloc( (void**) &d_dbdy_c,    GridSize );
    checkForCudaErrors("Could not allocate memory on level!",0);

    cudaMalloc( (void**) &d_Tbx,    GridSize );
    cudaMalloc( (void**) &d_Tby,    GridSize );
    cudaMalloc( (void**) &d_Tbz,    GridSize );
    cudaMalloc( (void**) &d_Ts,     GridSize );
    cudaMalloc( (void**) &d_Tn,     GridSize );
    cudaMalloc( (void**) &d_Vb,     GridSize );
    cudaMalloc( (void**) &d_VbRes,  GridSize );
    cudaMalloc( (void**) &d_beta,  GridSize );

    cudaMalloc( (void**) &d_Pw,     GridSize );
    cudaMalloc( (void**) &d_Pw_old,     GridSize );
    cudaMalloc( (void**) &d_PwSuggested,     GridSize );
    cudaMalloc( (void**) &d_PwR,     GridSize );
    cudaMalloc( (void**) &d_PwRes,     GridSize );
    checkForCudaErrors("Could not allocate memory on level for sliding!",0);

    cudaMalloc( (void**) &d_Scx,    GridSize );
    cudaMalloc( (void**) &d_ScxR,    GridSize );
    cudaMalloc( (void**) &d_ScxRes,    GridSize );
    cudaMalloc( (void**) &d_ScxSuggested,    GridSize );

    cudaMalloc( (void**) &d_Scy,    GridSize );
    cudaMalloc( (void**) &d_ScyR,    GridSize );
    cudaMalloc( (void**) &d_ScyRes,    GridSize );
    cudaMalloc( (void**) &d_ScySuggested,    GridSize );

    cudaMalloc( (void**) &d_hw,     GridSize );
    cudaMalloc( (void**) &d_hw_old,     GridSize );
    cudaMalloc( (void**) &d_hwR,     GridSize );
    cudaMalloc( (void**) &d_hwRes,     GridSize );
    cudaMalloc( (void**) &d_hwSuggested,     GridSize );

    cudaMalloc( (void**) &d_qsx,    GridSize );
    cudaMalloc( (void**) &d_qsy,    GridSize );
    cudaMalloc( (void**) &d_Qcx,    GridSize );
    cudaMalloc( (void**) &d_Qcy,    GridSize );
    cudaMalloc( (void**) &d_R,      GridSize );
    cudaMalloc( (void**) &d_Psi,    GridSize );
    cudaMalloc( (void**) &d_Mcx,    GridSize );
    cudaMalloc( (void**) &d_Mcy,    GridSize );

    cudaMalloc( (void**) &d_ResidualVec,  YNum*sizeof(ValueType) );  // Temp vector to hold partial residuales
    cudaMalloc( (void**) &d_MaxResVec,  YNum*sizeof(ValueType) );  // Temp vector to hold max values of residual
    checkForCudaErrors("Could not allocate memory for temp residual vector!",0);
  #else
    // If CPU version set d_* to point to host memory
    std::cout << "Setting CPU pointers" << std::endl;
    d_IceTopo = IceTopo;
    d_IceTopom= IceTopom;
    d_BedTopo = BedTopo;
    d_dhdx = dhdx;
    d_dhdy = dhdy;
    d_dbdx = dbdx;
    d_dbdy = dbdy;

    d_dhdx_c = dhdx_c;
    d_dhdy_c = dhdy_c;
    d_dbdx_c = dbdx_c;
    d_dbdy_c = dbdy_c;

    d_Sxx     = Sxx;
    d_Syy     = Syy;
    d_Sxy     = Sxy;

    d_Sxxm    = Sxxm;
    d_Syym    = Syym;
    d_Sxym    = Sxym;

    // This should be seperate memory from host values
    d_ResTaueNorm = (ValueType*) malloc(sizeof(ValueType));
    d_ResVxNorm   = (ValueType*) malloc(sizeof(ValueType));
    d_ResVyNorm   = (ValueType*) malloc(sizeof(ValueType));
    d_ResidualVec = (ValueType*) malloc(sizeof(ValueType)*YNum);
    d_MaxResVec   = (ValueType*) malloc(sizeof(ValueType)*YNum);
    
    d_Vxs      = Vxs;
    d_Vys      = Vys;

    d_Vxb      = Vxb;
    d_VxbSuggested      = VxbSuggested;
    d_VxbR   = VxbR;
    d_VxbRes = VxbRes;
    d_Vyb      = Vyb;
    d_VybSuggested      = VybSuggested;
    d_VybR   = VybR;
    d_VybRes = VybRes;

    d_Vxd      = Vxd;
    d_Vyd      = Vyd;

    d_Vxc      = Vxc;
    d_Vyc      = Vyc;

    d_Vx      = Vx;
    d_Vy      = Vy;
    d_VxRes   = VxRes;
    d_VyRes   = VyRes;
    d_VxSuggested = VxSuggested;
    d_VySuggested = VySuggested;
    d_VxR = VxR;
    d_VyR = VyR;
    d_Exx     = Exx;
    d_Eyy     = Eyy;
    d_Exy     = Exy;
    d_Exyc    = Exyc;
    d_Ezz     = Exx;
    d_Taue    = Taue;
    d_TaueRes = TaueRes;
    d_TaueSuggested = TaueSuggested;
    d_TaueR = TaueR;


    d_Tbx = Tbx;
    d_Tby = Tby;
    d_Tbz = Tbz;
    d_Ts  = Ts;
    d_Tn  = Tn;
    d_Vb  = Vb;
    d_beta = beta;
    d_VbRes  = VbRes;
    d_Pw  = Pw;
    d_Pw_old = Pw_old;
    d_PwR  = PwR;
    d_PwRes  = PwRes;
    d_PwSuggested  = PwSuggested;

    // Hydrology variables
    d_Scx = Scx;
    d_ScxR = ScxR;
    d_ScxRes = ScxRes;
    d_ScxSuggested = ScxSuggested;

    d_Scy = Scy;
    d_ScyR = ScyR;
    d_ScyRes = ScyRes;
    d_ScySuggested = ScySuggested;

    d_hw = hw;
    d_hw_old = hw_old;
    d_hwR = hwR;
    d_hwRes = hwRes;
    d_hwSuggested = hwSuggested;

    d_qsx = qsx;
    d_qsy = qsy;
    d_Qcx = Qcx;
    d_Qcy = Qcy;
    d_R = R;
    d_Psi = Psi;
    d_Mcx = Mcx;
    d_Mcy = Mcy;

  #endif

    
#ifdef __CPU__
    // Reset all arrays before we begin 
    
    std::cout << "Setting variables in RAM to zero" << std::endl;
    // Surface and bed velocivety
    memset (d_Vxs, 0.0, GridSize);
    memset (d_Vys, 0.0, GridSize);
    memset (d_Vxb, 0.0, GridSize);
    memset (d_Vyb, 0.0, GridSize);
    memset (d_VxbSuggested, 0.0, GridSize);
    memset (d_VybSuggested, 0.0, GridSize);
    memset (d_VxbR, 0.0, GridSize);
    memset (d_VybR, 0.0, GridSize);
    memset (d_VxbRes, 0.0, GridSize);
    memset (d_VybRes, 0.0, GridSize);

    memset (d_IceTopo, 0.0, GridSize);
    memset (d_IceTopom, 0.0, GridSize);
    memset (d_BedTopo, 0.0, GridSize);

    memset (d_Vxd, 0.0, GridSize);
    memset (d_Vyd, 0.0, GridSize);

    memset (d_Vxc, 0.0, GridSize);
    memset (d_Vyc, 0.0, GridSize);

    // Stresses
    memset (d_Sxx, 0.0, GridSize);
    memset (d_Syy, 0.0, GridSize);
    memset (d_Sxy, 0.0, GridSize);
    memset (d_Sxxm, 0.0, GridSize);
    memset (d_Syym, 0.0, GridSize);
    memset (d_Sxym, 0.0, GridSize);

    // Strain rates
    memset (d_Exx, 0.0, GridSize);
    memset (d_Eyy, 0.0, GridSize);
    memset (d_Exy, 0.0, GridSize);
    memset (d_Exyc, 0.0, GridSize);
    memset (d_Ezz, 0.0, GridSize);

    // Effective stress
    memset (d_Taue, 0.0, GridSize);
    memset (d_TaueR, 0.0, GridSize);
    memset (d_TaueRes, 0.0, GridSize);
    memset (d_TaueSuggested, 0.0, GridSize);

    // Vx velocity
    memset (d_Vx, 0.0, GridSize);
    memset (d_VxR, 0.0, GridSize);
    memset (d_VxRes, 0.0, GridSize);
    memset (d_VxSuggested, 0.0, GridSize);

    // Vy velocity
    memset (d_Vy, 0.0, GridSize);
    memset (d_VyR, 0.0, GridSize);
    memset (d_VyRes, 0.0, GridSize);
    memset (d_VySuggested, 0.0, GridSize);

    // Hydrology variables
    memset (d_Scx, 0.0, GridSize);
    memset (d_ScxR, 0.0, GridSize);
    memset (d_ScxRes, 0.0, GridSize);
    memset (d_ScxSuggested, 0.0, GridSize);

    memset (d_Scy, 0.0, GridSize);
    memset (d_ScyR, 0.0, GridSize);
    memset (d_ScyRes, 0.0, GridSize);
    memset (d_ScySuggested, 0.0, GridSize);
	
    memset (d_hw, 0.0, GridSize);
    memset (d_hw_old, 0.0, GridSize);
    memset (d_hwR, 0.0, GridSize);
    memset (d_hwRes, 0.0, GridSize);
    memset (d_hwSuggested, 0.0, GridSize);

    memset (d_Pw, 0.0, GridSize);
    memset (d_Pw_old, 0.0, GridSize);
    memset (d_PwR, 0.0, GridSize);
    memset (d_PwRes, 0.0, GridSize);
    memset (d_PwSuggested, 0.0, GridSize);

    memset (d_qsx, 0.0, GridSize);
    memset (d_qsy, 0.0, GridSize);
    memset (d_Qcx, 0.0, GridSize);
    memset (d_Qcy, 0.0, GridSize);
    memset (d_R, 0.1, GridSize);
    memset (d_Psi, 0.0, GridSize);
    memset (d_Mcx, 0.0, GridSize);
    memset (d_Mcy, 0.0, GridSize);

    memset (d_dhdx_c, 0.0, GridSize);
    memset (d_dhdy_c, 0.0, GridSize);
    memset (d_dbdx_c, 0.0, GridSize);
    memset (d_dbdy_c, 0.0, GridSize);

    // Clear "Device" residual values
    memset(d_ResTaueNorm,   0.0, sizeof(ValueType));
    memset(d_ResVxNorm,     0.0, sizeof(ValueType));
    memset(d_ResVyNorm,     0.0, sizeof(ValueType));
    memset(d_ResidualVec,   0.0, sizeof(ValueType)*YNum);
    memset(d_MaxResVec,        0.0, sizeof(ValueType)*YNum);
    std::cout << "Done" << std::endl;
    checkForCudaErrors("Reset of level to zero during init of level",1); // Note: Pause after reset
    #endif
    
    #ifdef __GPU__
    checkForCudaErrors("Before cudaMemcpy",1); // Note: Pause after reset

    cudaMemcpy(  d_Vxs,   Vxs,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vys,   Vys,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vxc,   Vxc,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vyc,   Vyc,  GridSize, cudaMemcpyHostToDevice );

    cudaMemcpy(  d_Vxb,   Vxb,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vyb,   Vyb,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxbSuggested,   VxbSuggested,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VybSuggested,   VybSuggested,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxbR,   VxbR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VybR,   VybR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxbRes,   VxbRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VybRes,   VybRes,  GridSize, cudaMemcpyHostToDevice );

    cudaMemcpy(  d_Vxd,   Vxd,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vyd,   Vyd,  GridSize, cudaMemcpyHostToDevice );

    // Copy gradients
    cudaMemcpy(  d_dhdx,   dhdx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dbdx,   dbdx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dhdy,   dhdy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dbdy,   dbdy,  GridSize, cudaMemcpyHostToDevice );

    cudaMemcpy(  d_dhdx_c,   dhdx_c,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dbdx_c,   dbdx_c,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dhdy_c,   dhdy_c,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_dbdy_c,   dbdy_c,  GridSize, cudaMemcpyHostToDevice );


    cudaMemcpy(  d_IceTopom,   IceTopom,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_BedTopo,   BedTopo,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_IceTopo,   IceTopo,  GridSize, cudaMemcpyHostToDevice );
 
    cudaMemcpy(  d_Sxx,   Sxx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Syy,   Syy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Sxy,   Sxy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Sxxm,   Sxxm,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Syym,   Syym,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Sxym,   Sxym,  GridSize, cudaMemcpyHostToDevice );

    cudaMemcpy(  d_Exx,   Exx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Eyy,   Eyy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Exy,   Exy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Exyc,   Exyc,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Ezz,   Ezz,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Taue,  Taue,  GridSize, cudaMemcpyHostToDevice );
 
    cudaMemcpy(  d_TaueR,  TaueR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxR,  VxR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VyR,  VyR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_TaueRes,  TaueRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_TaueSuggested,  TaueSuggested,  GridSize, cudaMemcpyHostToDevice );
 
    cudaMemcpy(  d_Vx,  Vx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_Vy,  Vy,  GridSize, cudaMemcpyHostToDevice );
 
    cudaMemcpy(  d_VxRes,  VxRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VyRes,  VyRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxR,  VxR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VyR,  VyR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VxSuggested,  VxSuggested,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy(  d_VySuggested,  VySuggested,  GridSize, cudaMemcpyHostToDevice );

    // Hydrology
    cudaMemcpy( d_Scx, Scx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_ScxR, ScxR,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_ScxRes, ScxRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_ScxSuggested, ScxSuggested,  GridSize, cudaMemcpyHostToDevice );

    cudaMemcpy( d_Scy, Scy,  GridSize, cudaMemcpyHostToDevice );   
    cudaMemcpy( d_ScyRes, ScyRes,  GridSize, cudaMemcpyHostToDevice );   
    cudaMemcpy( d_ScyR, ScyR,  GridSize, cudaMemcpyHostToDevice );   
    cudaMemcpy( d_ScySuggested, ScySuggested,  GridSize, cudaMemcpyHostToDevice );   

    cudaMemcpy( d_hw, hw,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_hw_old, hw_old,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_hwR, hwR,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_hwRes, hwRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_hwSuggested, hwSuggested,  GridSize, cudaMemcpyHostToDevice );  

    cudaMemcpy( d_Pw, Pw,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_Pw_old, Pw_old,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_PwR, PwR,  GridSize, cudaMemcpyHostToDevice );  
    cudaMemcpy( d_PwRes, PwRes,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_PwSuggested, PwSuggested,  GridSize, cudaMemcpyHostToDevice );  

    cudaMemcpy( d_qsx, qsx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_qsy, qsy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_Qcx, Qcx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_Qcy, Qcy,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_R, R,  GridSize, cudaMemcpyHostToDevice );    
    cudaMemcpy( d_Psi, Psi,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_Mcx, Mcx,  GridSize, cudaMemcpyHostToDevice );
    cudaMemcpy( d_Mcy, Mcy,  GridSize, cudaMemcpyHostToDevice );

    checkForCudaErrors("Coping zero arrays to GPU",1); // Note: Pause after reset
#endif
    //    cudaMemcpy(  Vxs,   d_Vxs,  GridSize, cudaMemcpyDeviceToHost );
    //    Print(Vxs);
    //    exit(0);

};

iSOSIA::~iSOSIA() {

  // TODO: Update with new arrays VxR, VxSuggested....

  delete[](IceTopo);
  delete[](BedTopo);
  delete[](Sxx); 
  delete[](Syy);
  delete[](Sxy);
  delete[](Vx);
  delete[](Vy);
  delete[](VxRes);
  delete[](VyRes);
  delete[](Exx);
  delete[](Eyy);
  delete[](Exy);
  delete[](Exyc);
  delete[](Ezz);
  delete[](Taue);
  delete[](TaueRes);
  //  delete[](ResidualVec);
  delete[](MaxResVec);

  #ifdef __GPU__
    // De-allocate Device memory
    cudaFree( d_IceTopo);
    cudaFree( d_BedTopo);
    cudaFree( d_Sxx); 
    cudaFree( d_Syy);
    cudaFree( d_Sxy);
    cudaFree( d_Vx);
    cudaFree( d_Vy);
    cudaFree( d_VxRes);
    cudaFree( d_VyRes);
    cudaFree( d_Exx);
    cudaFree( d_Eyy);
    cudaFree( d_Exy);
    cudaFree( d_Exyc);
    cudaFree( d_Ezz);
    cudaFree( d_Taue);
    cudaFree( d_TaueRes);
    cudaFree( d_ResidualVec);
    cudaFree( d_MaxResVec);
  #else
    delete[](d_IceTopo);
    delete[](d_BedTopo);
    delete[](d_Sxx); 
    delete[](d_Syy);
    delete[](d_Sxy);
    delete[](d_Vx);
    delete[](d_Vy);
    delete[](d_VxRes);
    delete[](d_VyRes);
    delete[](d_Exx);
    delete[](d_Eyy);
    delete[](d_Exy);
    delete[](d_Exyc);
    delete[](d_Ezz);
    delete[](d_Taue);
    delete[](d_TaueRes);
    delete[](d_ResidualVec);
  #endif

};

/*!
  This function changes the pointers used by SSIM to point to memory declared in SPM.
  This will result in unused memory haning around! If a memory leak should develop it will be here!
*/
void iSOSIA::FixSPMPointers(ValueType *SPMVx, ValueType *SPMVy, ValueType *SPMVxs, ValueType *SPMVys, 
			    ValueType *SPMVxb, ValueType *SPMVyb,
			    ValueType *SPMSxx, ValueType *SPMSyy, ValueType *SPMSxy, ValueType *SPMSzz, ValueType *SPMTaue, 
			    ValueType *SPMExx, ValueType *SPMEyy, ValueType *SPMExy, ValueType *SPMEzz, 
			    ValueType *SPMIceTopo, ValueType *SPMBedTopo, ValueType *SPMPw,
			    ValueType *SPMTbx, ValueType *SPMTby, ValueType *SPMTbz, ValueType *SPMTs, ValueType *SPMTn, ValueType *SPMVb, ValueType *SPMbeta,
			    ValueType *SPMdhdx, ValueType *SPMdhdy, ValueType *SPMdbdx, ValueType *SPMdbdy,
			    ValueType *SPMScx, ValueType *SPMScy, ValueType *SPMhw, ValueType *SPMqsx, ValueType *SPMqsy,
			    ValueType *SPMQcx, ValueType *SPMQcy, ValueType *SPMR, ValueType *SPMPsi, ValueType *SPMMcx, ValueType *SPMMcy
			    ) {

  // Hydrology
  Scx = SPMScx;
  Scy = SPMScy;
  hw = SPMhw;
  qsx = SPMqsx;
  qsy = SPMqsy;
  Qcx = SPMQcx;
  Qcy = SPMQcy;
  R = SPMR;
  Psi = SPMPsi;
  Mcx = SPMMcx;
  Mcy = SPMMcy;

  
  Vx = SPMVx;
  Vy = SPMVy;
  Vxs = SPMVxs;
  Vys = SPMVys;

  Vxb = SPMVxb;
  Vyb = SPMVyb;

  Sxx = SPMSxx;
  Syy = SPMSyy;
  Sxy = SPMSxy;
  //  Szz = SPMSzz;
  Exx = SPMExx;
  Eyy = SPMEyy;
  Exy = SPMExy;
  Ezz = SPMEzz;
  Taue = SPMTaue;
 
  // Topography pointes
  IceTopo = SPMIceTopo;
  BedTopo = SPMBedTopo;

  // Topo gradients
  dhdx = &SPMdhdx[0];
  dhdy = &SPMdhdy[0];
  dbdx = &SPMdbdx[0];
  dbdy = &SPMdbdy[0];

  // Sliding pointers
  Tbx = SPMTbx;
  Tby = SPMTby;
  Tbz = SPMTbz;
  Ts = SPMTs;
  Tn = SPMTn;
  Vb = SPMVb;
  beta = SPMbeta;
  Pw = SPMPw;

#ifndef __GPU__
  // It is important to reset the "device" pointers when
  // using the CPU and SPM Otherwise the above will override
  // the host pointers and device pointers will be NULL

  d_IceTopo = IceTopo;
  d_BedTopo = BedTopo;

  // Topo gradients
  // Note: Remove when switching to GPU
  d_dhdx_c = dhdx;
  d_dhdy_c = dhdy;
  d_dbdx_c = dbdx;
  d_dbdy_c = dbdy;


  d_Sxx     = Sxx;
  d_Syy     = Syy;
  d_Sxy     = Sxy;
  
  d_Vxs      = Vxs;
  d_Vys      = Vys;
  
  d_Vxb      = Vxb;
  d_Vyb      = Vyb;
  
  d_Vx      = Vx;
  d_Vy      = Vy;

  d_Exx     = Exx;
  d_Eyy     = Eyy;
  d_Exy     = Exy;
  d_Ezz     = Exx;
  d_Taue    = Taue;  
  
  d_Tbx = Tbx;
  d_Tby = Tby;
  d_Tbz = Tbz;
  d_Ts  = Ts;
  d_Tn  = Tn;
  d_Vb  = Vb;
  d_beta  = beta;
  d_Pw  = Pw;

  // Hydrology variables
  d_Scx = Scx;
  d_Scy = Scy;
  d_hw = hw;
  d_qsx = qsx;
  d_qsy = qsy;
  d_Qcx = Qcx;
  d_Qcy = Qcy;
  d_R = R;
  d_Psi = Psi;
  d_Mcx = Mcx;
  d_Mcy = Mcy;

#endif

};

void iSOSIA::setTestATopo(ValueType XStp, ValueType YStp, ValueType XSize, ValueType YSize) {

  #ifdef __GPU__
  dim3 GPUGridSizeTopo = this->GPUGridSize;
  GPUGridSizeTopo.x += 1;
  GPUGridSizeTopo.y += 1;
  g_Compute_test_topography <<<GPUGridSizeTopo, this->GPUBlockSize>>> (d_IceTopo, d_BedTopo,
								       d_dbdx_c, d_dbdy_c,
								       d_dhdx_c, d_dhdy_c,
								       d_dbdx, d_dbdy,
								       d_dhdx, d_dhdy,
								       0, LevelNum);
  #else
  g_Compute_test_topography  (d_IceTopo, d_BedTopo,
			      d_dbdx_c, d_dbdy_c,
			      d_dhdx_c, d_dhdy_c,
			      d_dbdx, d_dbdy,
			      d_dhdx, d_dhdy,
			      0, LevelNum);
#endif
  checkForCudaErrors("Computing topograhpy", 1);
    
};

void iSOSIA::FillGhost(ValueType *M, const int XStart, const int YStart, const int mode) {
  // <<<GPUGridSize,GPUBlockSize>>>
  #ifdef __GPU__
  dim3 GPUGridSizeTopo = this->GPUGridSize;
  GPUGridSizeTopo.x += 1;
  GPUGridSizeTopo.y += 1;
  
  fillGhost  <<<GPUGridSizeTopo, this->GPUBlockSize>>> ( M, YStart, YNum-2, 
			  XStart, XNum-2, 
			  LevelNum, mode);

 #else
  fillGhost ( M, YStart, YNum-2, 
	      XStart, XNum-2, 
	      LevelNum, mode);
#endif
  checkForCudaErrors("Filling ghost nodes", 1); 
};


void iSOSIA::ComputeErrorForProlong(const ValueType* u, const ValueType* v, ValueType *error, const int xShift, const int yShift) {
  //ResetLevel DEVICESPECS( error, 0.0, LevelNum);
  ComputeError DEVICESPECS(u, v, error, LevelNum, xShift, yShift);

};

void iSOSIA::RestrictTopography(iSOSIA* FineLevel, iSOSIA* CoarseLevel) {

  // Restricts topography to coarse grids
  printf("Restricting topography level %i -> %i \n",FineLevel->LevelNum,CoarseLevel->LevelNum);
  printf("From XNum %i YNum %i \n",FineLevel->XNum,FineLevel->YNum);
  printf("To   XNum %i YNum %i \n",CoarseLevel->XNum,CoarseLevel->YNum);

  // Restriction save fine grid interpolation to coarse grid
  #ifdef __GPU__
  restriction<<<CoarseLevel->GPUGridSize,CoarseLevel->GPUBlockSize>>>(FineLevel->d_IceTopo, CoarseLevel->d_IceTopo, 0.5, 0.5, 0, FineLevel->d_IceTopo, CoarseLevel->d_IceTopo, CoarseLevel->LevelNum, 0, 0); // Note Last two IceTopo grids are just to keep the structure
  restriction<<<CoarseLevel->GPUGridSize,CoarseLevel->GPUBlockSize>>>(FineLevel->d_BedTopo, CoarseLevel->d_BedTopo, 0.5, 0.5, 0, FineLevel->d_BedTopo, CoarseLevel->d_BedTopo, CoarseLevel->LevelNum, 0, 0);  
  checkForCudaErrors("Restricting topography", 1); 
  
  fillGhost<<<CoarseLevel->GPUGridSize,CoarseLevel->GPUBlockSize>>>( CoarseLevel->d_BedTopo, 1, CoarseLevel->YNum-2, 1, CoarseLevel->XNum-2, CoarseLevel->LevelNum, 1);
  fillGhost<<<CoarseLevel->GPUGridSize,CoarseLevel->GPUBlockSize>>>( CoarseLevel->d_IceTopo, 1, CoarseLevel->YNum-2, 1, CoarseLevel->XNum-2, CoarseLevel->LevelNum, 1);
  checkForCudaErrors("Filling ghost nodes for bed and ice topography", 1); 
  
  #else
  restriction(FineLevel->d_IceTopo, CoarseLevel->d_IceTopo, 
	      0.5, 0.5, 0, 
	      FineLevel->d_IceTopo, CoarseLevel->d_IceTopo, 
	      CoarseLevel->LevelNum, 0, 0); // Note Last two IceTopo grids are just to keep the structure
  
  restriction(FineLevel->d_BedTopo, CoarseLevel->d_BedTopo, 
	      0.5, 0.5, 0, 
	      FineLevel->d_BedTopo, CoarseLevel->d_BedTopo, 
	      CoarseLevel->LevelNum, 0, 0);

  fillGhost( CoarseLevel->d_BedTopo, 1, CoarseLevel->YNum-2, 1, CoarseLevel->XNum-2, CoarseLevel->LevelNum, 1);
  fillGhost( CoarseLevel->d_IceTopo, 1, CoarseLevel->YNum-2, 1, CoarseLevel->XNum-2, CoarseLevel->LevelNum, 1);

  #endif

};

/**
   This function copies memory from the GPU to
   host memory.

   Note: Is always called but only preforms a copy 
         when GPU flag is set.
 */
void iSOSIA::CopyLevelToDevice() { 
  #ifdef __GPU__
  cudaMemcpy( d_IceTopo, IceTopo,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_BedTopo, BedTopo,  GridSize, cudaMemcpyHostToDevice ); 
  //  cudaMemcpy( d_R, R,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_Pw, Pw,  GridSize, cudaMemcpyHostToDevice );

  //  if (CopyGradients) {
  cudaMemcpy( d_dbdx, dbdx,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_dbdy, dbdy,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_dhdx, dhdx,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_dhdy, dhdy,  GridSize, cudaMemcpyHostToDevice );

    dim3 GPUGridSizeTopo = this->GPUGridSize;
  GPUGridSizeTopo.x += 1;
  GPUGridSizeTopo.y += 1;

  cuInterpGradients <<<GPUGridSizeTopo, 
    this->GPUBlockSize>>> (d_dhdx, d_dhdy, 
			   d_dbdx, d_dbdy,
			   d_dhdx_c, d_dhdy_c, 
			   d_dbdx_c, d_dbdy_c,
			   LevelNum);
  
  SetBoundaries(4);    
  
/*
  cudaMemcpy( d_hw,    hw,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_Scx,   Scx,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_Scy,   Scy,  GridSize, cudaMemcpyHostToDevice ); 
  cudaMemcpy( d_Psi,   Psi,  GridSize, cudaMemcpyHostToDevice ); 
*/
  checkForCudaErrors("Copying level to device", 1);
  printf("Level copied to device");
  #endif
};

/**
   This function copies memory from the host to
   GPU memory.

   Note: Is always called but only preforms a copy 
         when GPU flag is set.
 */
void iSOSIA::CopyLevelToHost( const bool FileIO) {
  #ifdef __GPU__

  cudaMemcpy(  Vx,    d_Vx,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Vy,    d_Vy,  GridSize, cudaMemcpyDeviceToHost ); 
  
  cudaMemcpy(  Vxd,    d_Vxd,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Vyd,    d_Vyd,  GridSize, cudaMemcpyDeviceToHost ); 
  
  cudaMemcpy(  Vxs,    d_Vxs,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Vys,    d_Vys,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Vxb,    d_Vxb,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Vyb,    d_Vyb,  GridSize, cudaMemcpyDeviceToHost ); 

  /*
  cudaMemcpy(  Pw,    d_Pw,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  hw,    d_hw,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  qsx,    d_qsx,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  qsy,    d_qsy,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Psi,    d_Psi,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  R,    d_R,  GridSize, cudaMemcpyDeviceToHost ); 
  */

  cudaMemcpy(  Ts,  d_Ts, GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy(  Tn,  d_Tn, GridSize, cudaMemcpyDeviceToHost ); 

  /*
  cudaMemcpy( dbdx, d_dbdx,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy( dbdy, d_dbdy,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy( dhdx, d_dhdx,  GridSize, cudaMemcpyDeviceToHost ); 
  cudaMemcpy( dhdy, d_dhdy,  GridSize, cudaMemcpyDeviceToHost );
  */
  
  if (FileIO) {
    // Only copy to host if they are written to file
    cudaMemcpy(  Sxx,   d_Sxx,  GridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy(  Syy,   d_Syy,  GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Sxy,   d_Sxy,  GridSize, cudaMemcpyDeviceToHost ); 
    
    cudaMemcpy(  Sxxm,  d_Sxxm, GridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy(  Syym,  d_Syym, GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Sxym,  d_Sxym, GridSize, cudaMemcpyDeviceToHost ); 
    
    cudaMemcpy(  Exx,   d_Exx,  GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Eyy,   d_Eyy,  GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Exy,   d_Exy,  GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Ezz,   d_Ezz,  GridSize, cudaMemcpyDeviceToHost ); 
    cudaMemcpy(  Taue,  d_Taue, GridSize, cudaMemcpyDeviceToHost );
    cudaMemcpy(  beta,  d_beta, GridSize, cudaMemcpyDeviceToHost );
    
  }


  std::cout << "Normal Variables copied to host" << std::endl;
  checkForCudaErrors("Copying level to host", 1);

  #endif
};

void iSOSIA::solveHydrology (double dt) {

  int hruns = 0;
  int pruns = 0;
  ValueType SRes = 1e10; // Residual on channel cross-section
  ValueType HRes = 1e10;
printf("\n");
  while ( 5000 > hruns && HRes > 1e-3 ) {
  //    SRes = 1e10;
    pruns = 0;
    SRes = 1;
//cloneArray(d_Pw, d_Pw_old);

while ( 1000 > pruns && SRes > 1e-3 ) {
    g_hydrology_update_psi DEVICESPECS (d_Psi, d_BedTopo, d_Pw, LevelNum);
    checkForCudaErrors("Hydrology first psi update ", 1);
    
    g_hydrology_update_flux DEVICESPECS (d_Qcx, d_Qcy, d_qsx, d_qsy,
					 d_Psi, d_Scx, d_hw,
					 d_IceTopo, d_BedTopo,
					 LevelNum, 0, 0);
    
    g_hydrology_update_flux DEVICESPECS (d_Qcx, d_Qcy, d_qsx, d_qsy,
					 d_Psi, d_Scx, d_hw,
					 d_IceTopo, d_BedTopo,
					 LevelNum, 1, 0);
    checkForCudaErrors("Hydrology flux update ", 1);
    
    //  printf("\n");


      /*
	g_hydrology_update_sx DEVICESPECS (d_Scx, d_ScxSuggested, d_ScxR, d_ScxRes,
	d_Mcx, d_Psi, d_Pw, d_Scx, d_Qcx,
	d_qsx, d_Tn, dt,
				       d_IceTopo, d_BedTopo,
				       LevelNum, 0, 0);
    g_hydrology_update_sx DEVICESPECS (d_Scx, d_ScxSuggested, d_ScxR, d_ScxRes,
				       d_Mcx, d_Psi, d_Pw, d_Scx, d_Qcx,
				       d_qsx, d_Tn, dt,
				       d_IceTopo, d_BedTopo,
				       LevelNum, 1, 0);
    g_hydrology_update_sy DEVICESPECS (d_Scy, d_ScySuggested, d_ScyR, d_ScyRes,
				       d_Mcy, d_Psi, d_Pw, d_Scy, d_Qcy,
				       d_qsy, d_Tn, dt,
				       d_IceTopo, d_BedTopo,
				       LevelNum, 0, 0);
    g_hydrology_update_sy DEVICESPECS (d_Scy, d_ScySuggested, d_ScyR, d_ScyRes,
				       d_Mcy, d_Psi, d_Pw, d_Scy, d_Qcy,
				       d_qsy, d_Tn, dt, 
				       d_IceTopo, d_BedTopo,
				       LevelNum, 1, 0);
    SRes = HydrologyResidual_s();
    if (SRes != 0.0)
      printf(" Channel Res = %f\n",SRes);

    checkForCudaErrors("Hydrology channel update", 1);
*/        
      g_hydrology_update_pressure DEVICESPECS (d_Pw, d_PwSuggested, d_PwR, d_PwRes, d_Pw_old,
    d_hw, d_Tn, 
    d_Ts, d_Vb,
    d_qsx, d_qsy,
    d_R, dt, 
    d_dbdx_c, d_dbdy_c,
    d_IceTopo, d_BedTopo,
    LevelNum, 0, 0);
      
      g_hydrology_update_pressure DEVICESPECS (d_Pw, d_PwSuggested, d_PwR, d_PwRes, d_Pw_old,
    d_hw, d_Tn, 
    d_Ts, d_Vb,
    d_qsx, d_qsy,
    d_R, dt, 
    d_dbdx_c, d_dbdy_c,
    d_IceTopo, d_BedTopo,
    LevelNum, 1, 0);
  checkForCudaErrors("Hydrology pressure update", 1);

  SRes = HydrologyResidual_s();
  if (SRes != 0.0 && 0)
    printf("\n Presure Res = %f",SRes);
  
  pruns++;
    }
  
    //  printf("\n");

  g_hydrology_update_psi DEVICESPECS (d_Psi, d_BedTopo, d_Pw, LevelNum);
  checkForCudaErrors("Hydrology second psi update ", 1);

    g_hydrology_update_flux DEVICESPECS (d_Qcx, d_Qcy, d_qsx, d_qsy,
					 d_Psi, d_Scx, d_hw,
					 d_IceTopo, d_BedTopo,
					 LevelNum, 0, 0);
    
    g_hydrology_update_flux DEVICESPECS (d_Qcx, d_Qcy, d_qsx, d_qsy,
					 d_Psi, d_Scx, d_hw,
					 d_IceTopo, d_BedTopo,
					 LevelNum, 1, 0);
    checkForCudaErrors("Hydrology flux update ", 1);

  g_hydrology_update_h DEVICESPECS (d_hw, d_hwSuggested, d_hwR, d_hwRes, d_hw_old,
    d_Pw_old, d_Pw,
				    d_Qcx, d_qsx, d_Qcy, d_qsx,
				    d_Psi, d_Mcx, d_Mcy, d_R, dt, d_Scx,
				    d_Ts, d_Vb, 
				    d_IceTopo, d_BedTopo,
				    LevelNum, 0, 0);
  g_hydrology_update_h DEVICESPECS (d_hw, d_hwSuggested, d_hwR, d_hwRes, d_hw_old,
    d_Pw_old, d_Pw,
				    d_Qcx, d_qsx, d_Qcy, d_qsx,
				    d_Psi, d_Mcx, d_Mcy, d_R, dt, d_Scx,
				    d_Ts, d_Vb, 
				    d_IceTopo, d_BedTopo,
				    LevelNum, 1, 0);

  //  printf("\n h Res");
  HRes = HydrologyResidual_h();
  if (HRes != 0.0/* && HRes > 1e-3*/ )
    printf("\r h Res = %f",HRes);

  checkForCudaErrors("Hydrology water depth!", 0);  
  hruns++;
  }
};

void iSOSIA::ComputeStrainRate() {
  SetBoundaries();

  g_Compute_Exy_cross_terms DEVICESPECS (d_Vx, d_Vy, d_Exyc, LevelNum);
  checkForCudaErrors("Post strain cross derivative kernel call level!", 1);

  // Set BC for Exym before the interpolation
  /*  #ifdef __CPU__
      SetBoundaries(1);
  #endif
  */

   g_Interp_Exy DEVICESPECS (d_Exyc, d_Exy, LevelNum);
  checkForCudaErrors("Post strain interp kernel call level!", 1);


  //SetBoundaries(2);
  
  // Compute strain rates
  g_Compute_strain_rate DEVICESPECS (d_Vx, d_Vy,
				     d_Vxs, d_Vys,
				     d_Vxb, d_Vyb,
				     d_Exx, d_Eyy, 
				     d_Exy, d_Ezz,
				     d_IceTopo, d_BedTopo,
				     d_dhdx_c, d_dhdy_c,
				     d_dbdx_c, d_dbdy_c,
				     LevelNum, 0);
  checkForCudaErrors("Post strain rate kernel call!", 1);

  //SetBoundaries(2);
    
  //  g_CenterToCornerInterp DEVICESPECS (d_Exy, d_Exyc, LevelNum);
  //  checkForCudaErrors("Post strain rate interpolation to corner nodes!", 1);

  SetBoundaries();
};

void iSOSIA::IterateStress(const int runs, const int updateWanted) {
  SetBoundaries();

  for (int run = 0; run < runs; run++) {
    g_update_stress DEVICESPECS (d_Sxx, d_Syy, d_Sxy, 
				 d_Exx, d_Eyy, d_Exy,
				 d_Taue, d_IceTopo, d_BedTopo,
				 LevelNum, 0);
        
    checkForCudaErrors("Red stress update", 1);
    SetBoundaries();

    g_update_stress DEVICESPECS (d_Sxx, d_Syy, d_Sxy, 
				 d_Exx, d_Eyy, d_Exy,
				 d_Taue, d_IceTopo, d_BedTopo,
				 LevelNum, 1);
    
    checkForCudaErrors("Black stress update", 1); // Ignored on CPU calls    
    SetBoundaries();

    // Iterate Stress
    g_update_effective_stress DEVICESPECS (d_Taue, d_TaueSuggested, d_TaueR, d_TaueRes,
					   d_Sxx, d_Syy, d_Sxy,
					   d_IceTopo, d_BedTopo, 
					   d_dhdx_c, d_dhdy_c,
					   LevelNum, 0, updateWanted); // Red
    
    checkForCudaErrors("Post coeffieient kernel Red call level!", 1);
    SetBoundaries();

    g_update_effective_stress DEVICESPECS (d_Taue, d_TaueSuggested, d_TaueR, d_TaueRes,
					   d_Sxx, d_Syy, d_Sxy,
					   d_IceTopo, d_BedTopo, 
					   d_dhdx_c, d_dhdy_c,
					   LevelNum, 1, updateWanted); // Black

    checkForCudaErrors("Post coeffieient kernel Black call level!", 1);

    SetBoundaries();  
  };
};

void iSOSIA::BlockIterateStress(const int color, const int updateWanted) {
  SetBoundaries();

    g_update_stress DEVICESPECS (d_Sxx, d_Syy, d_Sxy, 
				 d_Exx, d_Eyy, d_Exy,
				 d_Taue, d_IceTopo, d_BedTopo,
				 LevelNum, color);
        
    checkForCudaErrors("Red stress update", 1);
    SetBoundaries();

    // Iterate Stress
    g_update_effective_stress DEVICESPECS (d_Taue, d_TaueSuggested, d_TaueR, d_TaueRes,
					   d_Sxx, d_Syy, d_Sxy,
					   d_IceTopo, d_BedTopo, 
					   d_dhdx_c, d_dhdy_c,
					   LevelNum, color, updateWanted); // Red
    
    checkForCudaErrors("Post coeffieient kernel Red call level!", 1);
    SetBoundaries();
};

void iSOSIA::InterpIceTopographyToCorner() {
  g_CenterToCornerInterp DEVICESPECS (d_IceTopo, d_IceTopom, LevelNum);
};

void iSOSIA::getGradients() {
  
  #ifdef __GPU__
  dim3 GPUGridSizeTopo = this->GPUGridSize;
  GPUGridSizeTopo.x += 1;
  GPUGridSizeTopo.y += 1;
  

  cuGetGradients <<<GPUGridSizeTopo, 
    this->GPUBlockSize>>> (
			   d_IceTopo, d_BedTopo,
			   d_dhdx, d_dhdy, 
			   d_dbdx, d_dbdy, 
			   LevelNum);

  cuInterpGradients <<<GPUGridSizeTopo, 
                       this->GPUBlockSize>>> (d_dhdx, d_dhdy, 
					      d_dbdx, d_dbdy,
					      d_dhdx_c, d_dhdy_c, 
					      d_dbdx_c, d_dbdy_c,
					      LevelNum);
  
  SetBoundaries(4);    
    //FixVelocityGradients(3);
#else
  cuGetGradients (d_IceTopo, d_BedTopo, d_dhdx, 
		  d_dhdy, d_dbdx, d_dbdy, LevelNum);
  cuInterpGradients (d_dhdx, d_dhdy, d_dbdx, d_dbdy,
		     d_dhdx_c, d_dhdy_c, d_dbdx_c, d_dbdy_c,
		     LevelNum);

#endif
};

void iSOSIA::IterateVelocity(const int updateWanted, 			   
			     const bool PostProlongUpdate,
			     const int Blockwise) {

  // Interpolate values to corner nodes
  SetBoundaries(0);

  /*
  g_CenterToCornerInterp DEVICESPECS (d_Sxx, d_Sxxm, LevelNum);
  g_CenterToCornerInterp DEVICESPECS (d_Syy, d_Syym, LevelNum);
  g_CenterToCornerInterp DEVICESPECS (d_Sxy, d_Sxym, LevelNum);
  checkForCudaErrors("Post interpolation of center stress nodes to corner!", 1);  
  SetBoundaries(0);
  */
  g_update_vx DEVICESPECS (d_Vxd, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, 0, 0, updateWanted, PostProlongUpdate, Blockwise); // Red - Vx
  
  checkForCudaErrors("Post velocity update - Red Vx!", 1);
  SetBoundaries(0);

  g_update_vx DEVICESPECS (d_Vxd, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, 1, 0, updateWanted, PostProlongUpdate, Blockwise); // Black - Vx
  
  checkForCudaErrors("Post velocity update - Black Vx!", 1);
  SetBoundaries(0);
 
  g_update_vy DEVICESPECS (d_Vyd, d_VySuggested, d_VyR, d_VyRes,
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, 0, 0, updateWanted, PostProlongUpdate, Blockwise); // Red - Vy
  
  checkForCudaErrors("Post velocity update - Red Vy!", 1);
  SetBoundaries(0);

  g_update_vy DEVICESPECS (d_Vyd, d_VySuggested, d_VyR, d_VyRes, 
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, 1, 0, updateWanted, PostProlongUpdate, Blockwise); // Black - Vy
  
  checkForCudaErrors("Post velocity update - Black Vy!", 1);
  SetBoundaries(0);  

};

void iSOSIA::BlockIterateVelocity(const int updateWanted, 			   
				  const bool PostProlongUpdate,
				  const int color,
				  const int Blockwise
				  ) {

  // Interpolate values to corner nodes
  SetBoundaries(0);

  g_update_vx DEVICESPECS (d_Vxd, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, color, 0, updateWanted, PostProlongUpdate, Blockwise); // Red - Vx
  
  checkForCudaErrors("Post velocity update - Red Vx!", 1);
  SetBoundaries(0);
 
  g_update_vy DEVICESPECS (d_Vyd, d_VySuggested, d_VyR, d_VyRes,
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, color, 0, updateWanted, PostProlongUpdate, Blockwise); // Red - Vy
  
  checkForCudaErrors("Post velocity update - Red Vy!", 1);
  SetBoundaries(0);
};

void iSOSIA::ComputeSlidingVelocity (const int Blockwise) {

  // Red Update
  g_compute_sliding_terms DEVICESPECS (d_Tbx, d_Tby, d_Tbz, d_Vb, d_Ts, d_Tn, d_beta,
				       d_Sxx, d_Syy, d_Sxy,
				       d_Pw, d_IceTopo, d_BedTopo, 
				       d_dhdx_c, d_dhdy_c, d_dbdx_c, d_dbdy_c,
				       LevelNum, 0);
  checkForCudaErrors("Post sliding term computation!", 1);
  SetBoundaries();

  // Black Update
  g_compute_sliding_terms DEVICESPECS (d_Tbx, d_Tby, d_Tbz, d_Vb, d_Ts, d_Tn, d_beta,
				       d_Sxx, d_Syy, d_Sxy,
				       d_Pw, d_IceTopo, d_BedTopo, 
				       d_dhdx_c, d_dhdy_c, d_dbdx_c, d_dbdy_c,
				       LevelNum, 1);
  checkForCudaErrors("Post sliding term computation!", 1);
  SetBoundaries();

  g_update_vxb DEVICESPECS (d_Vxb, d_VxbSuggested, d_VxbR, d_VxbRes,
			    d_Vxd, d_Vx,
			    d_Tbx, d_Ts, d_Vb, 
			    d_IceTopo, d_BedTopo, 
			    LevelNum, 0, Blockwise);
  checkForCudaErrors("Post Vxb sliding update!", 1);
  //SetBoundaries();

  g_update_vxb DEVICESPECS (d_Vxb, d_VxbSuggested, d_VxbR, d_VxbRes,
			    d_Vxd, d_Vx,
			    d_Tbx, d_Ts, d_Vb, 
			    d_IceTopo, d_BedTopo, 
			    LevelNum, 1, Blockwise);
  checkForCudaErrors("Post Vxb sliding update!", 1);
  //SetBoundaries();
  
  g_update_vyb DEVICESPECS (d_Vyb, d_VybSuggested, d_VybR, d_VybRes,
			    d_Vyd, d_Vy,
			    d_Tby, d_Ts, d_Vb, 
			    d_IceTopo, d_BedTopo, 
			    LevelNum, 0, Blockwise);
  checkForCudaErrors("Post Vyb sliding update!", 1);
  // SetBoundaries();
  
  g_update_vyb DEVICESPECS (d_Vyb, d_VybSuggested, d_VybR, d_VybRes,
			    d_Vyd, d_Vy,
			    d_Tby, d_Ts, d_Vb, 
			    d_IceTopo, d_BedTopo, 
			    LevelNum, 1, Blockwise);
  checkForCudaErrors("Post Vyb sliding update!", 1);
  // SetBoundaries();

  g_Compute_sliding_residual DEVICESPECS (d_VbRes, d_Vxb, d_Vyb, d_Vb, LevelNum);   
  checkForCudaErrors("Post sliding residual!", 1);
};

/**
 * This function computes the surface velocity.
 */
void iSOSIA::ComputeSurfaceVelocity(const int updateType) {
  SetBoundaries();    

  //
  // Compute surface velocities
  //
  g_update_vx DEVICESPECS (d_Vx, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, 0, updateType, 0, 0); // Red - Vx
  checkForCudaErrors("Post surface velocity update - Red Vx!", 0);
  SetBoundaries();
  
  g_update_vx DEVICESPECS (d_Vx, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, 1, updateType, 0, 0); // Black - Vx
  checkForCudaErrors("Post surface velocity update - Black Vx!", 0);

  SetBoundaries();    
  g_update_vy DEVICESPECS (d_Vy, d_VySuggested, d_VyR, d_VyRes,
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, 0, updateType, 0, 0); // Red - Vy
  checkForCudaErrors("Post surface velocity update - Red Vy!", 0);
  SetBoundaries();

  g_update_vy DEVICESPECS (d_Vy, d_VySuggested, d_VyR, d_VyRes,
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, 1, updateType, 0, 0); // Black - Vy
  checkForCudaErrors("Post surface velocity update - Black Vy!", 1);
  SetBoundaries();

};

/**
 * This function computes the surface velocity.
 */
void iSOSIA::BlockComputeSurfaceVelocity(const int color, const int updateType, const int Blockwise) {
  SetBoundaries();    

  //
  // Compute surface velocities
  //
  g_update_vx DEVICESPECS (d_Vx, d_VxSuggested, d_VxR, d_VxRes,
			   d_Vx, d_Vxs, d_Vxb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx, d_dhdy_c,
			   LevelNum, color, updateType, 0, 0, Blockwise); // Red - Vx
  checkForCudaErrors("Post surface velocity update - Red Vx!", 0);
  SetBoundaries();
  
  g_update_vy DEVICESPECS (d_Vy, d_VySuggested, d_VyR, d_VyRes,
			   d_Vy, d_Vys, d_Vyb, d_Ezz,
			   d_Sxx, d_Syy, d_Sxy,
			   d_Sxxm, d_Syym, d_Sxym,
			   d_IceTopo, d_IceTopom, d_BedTopo, 
			   d_dhdx_c, d_dhdy,
			   LevelNum, color, updateType, 0, 0, Blockwise); // Red - Vy
  checkForCudaErrors("Post surface velocity update - Red Vy!", 0);
  SetBoundaries();

};

/*
 * Computes the max/inf norm of a 2D matrix
 * 
 * @param d_Min Input matrix on device
 */
ValueType iSOSIA::ComputeMaxNorm(ValueType *d_Min) {
  
  ValueType MaxVal = 1.0e-16;
  //  printf("\n Computing Max Norm\n");

  memset (MaxResVec, 0.0, (YNum)*sizeof(ValueType));

  #ifdef __GPU__
  cudaMemcpy( d_MaxResVec, MaxResVec, YNum*sizeof(ValueType), cudaMemcpyHostToDevice);
  #endif    
  checkForCudaErrors("After level reset function", 1);

  MaxRows DEVICESPECS (d_Min, d_MaxResVec, LevelNum);
  checkForCudaErrors("Max in matrix (Row)!", 1);
  //  printf("Max row found\n");

  #ifdef __GPU__
  cudaMemcpy( MaxResVec, d_MaxResVec, YNum*sizeof(ValueType), cudaMemcpyDeviceToHost);
  checkForCudaErrors("MemCpy of MaxVecRes from GPU!", 1);
  #else
  MaxResVec = (d_MaxResVec);
  #endif

  //  printf("Memory copied");
  for (int Row = 0; Row < YNum; Row++) {
    //    printf("Row %i = %f\n",Row,MaxResVec[Row]);
    if (MaxResVec[Row] > MaxVal && MaxResVec[Row] != 0.0 ) {
      MaxVal = MaxResVec[Row];
    }
  }

  return MaxVal;

}
  /**
   * Computes the Norm of a matrix and it's residual. Norm is defined as,
   *
   * R = |(f_new - f)^2|/(f_new)^2
   * 
   * or a norm where M is f_new - f and MNew is f_new (this is normType=1)
   *
   * @param M Device pointer to Matrix to compute residual from
   * @param MNew Device pointer to the new values suggested by iteration
   * @param xShift Shift start col by a integer
   * @param yShift Shift start row by a integer
   * @param normType The type of norm to perform. 0 is L2 while 1 is a simple mean
   */
ValueType iSOSIA::ComputeResidual(ValueType *M, ValueType *MNew, 
				  const unsigned int xShift, const unsigned int yShift,
				  const int normType = 0)  {
  ValueType SumDiff = 0.0;
  ValueType SumTot = 0.0;
  double ResResult = 0.0;

  #ifdef __GPU__
    ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum-1, 1, 0);
  #else
    memset (d_ResidualVec, 0.0, (YNum)*sizeof(ValueType));
  #endif
    
  checkForCudaErrors("After level reset function", 1);

  if (normType == 1) {
    // Compute a simple mean

    // Sum all rows to one vector
    SumRows DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 3, LevelNum);
    checkForCudaErrors("Sum over row in matrix!", 1);
    
    // Sum vector
    SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 
			 0, // xShift
			 -1, // yShift
			 1, // Dimensions
			 2, // Compute diff
			 LevelNum); // Compute only ResidualVec
    checkForCudaErrors("Sum over Residual vector!", 1);
  } else if (normType == 0) {
    // Compute L2Norm
    SumRows DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 1, LevelNum);
    checkForCudaErrors("Sum over row in matrix!", 1);
    
    SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 
			 0, // xShift
			 -1, // yShift
			 1, // Dimensions
			 2, // Compute diff
			 LevelNum); // Compute only ResidualVec
    checkForCudaErrors("Sum over Residual vector!", 1);
  }
  
  //  std::cout << "SumDiff: " << SumDiff << "\n" << std::endl;
  #ifdef __GPU__
  cudaMemcpy( &SumDiff, d_ResTaueNorm, sizeof(ValueType), cudaMemcpyDeviceToHost);
  checkForCudaErrors("MemCpy of SumDiff from GPU!", 1);
  #else
  SumDiff = (*d_ResTaueNorm);
  #endif

    // Compute L2Norm
#ifdef __GPU__
    ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum-1, 1, 0);
#else
    memset (d_ResidualVec, 0.0, (YNum)*sizeof(ValueType));
#endif
    
    checkForCudaErrors("After second level reset function", 1);
    
    SumRows DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 0, LevelNum);
    checkForCudaErrors("Sum over row in matrix!", 1);
    SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 0/*xShift*/, -1, 1, 2, LevelNum); // Compute only ResidualVec 
    // (i.e. sum over 0 row in ResidualVec)
    checkForCudaErrors("Sum over Residual vector!", 1);
    
    
#ifdef __GPU__
    cudaMemcpy( &SumTot, d_ResTaueNorm, sizeof(ValueType), cudaMemcpyDeviceToHost);
    checkForCudaErrors("MemCpy of SumTot from GPU!", 1);
#else
    SumTot = (*d_ResTaueNorm);
#endif

  ///  std::cout << "SumTot: " << SumTot << " SumDiff "<< SumDiff << std::endl;
  
    ResResult=(SumDiff)/(SumTot+10e-16);
  
  if (ResResult != ResResult) {
    dumpGPUArrayToFile(M, "M.data", LevelNum);
    dumpGPUArrayToFile(MNew, "MNew.data", LevelNum);
    std::cout << "The residual is NaN!" << std::endl;
    std::cout << "Residual: (SumDiff)/(SumTot+10e-16) = " << SumDiff << "/" << (SumTot+10e-16) << " = " << ResResult << std::endl;
    ResResult = 0.0;
    exit(1);
  };

  return ResResult;
};

ValueType iSOSIA::BlockComputeResidual(ValueType *M, ValueType *MNew, 
				       const unsigned int xShift, const unsigned int yShift,
				       const int color)  {
  ValueType SumDiff = 0.0;
  ValueType SumTot = 0.0;
  double ResResult = 0.0;

  #ifdef __GPU__
    ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum-1, 1, 0);
  #else
    memset (d_ResidualVec, 0.0, (YNum)*sizeof(ValueType));
  #endif
    
  checkForCudaErrors("After level reset function", 1);

  // Compute L2Norm
  SumRowsBlockwise DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 1, LevelNum, color);
  checkForCudaErrors("Sum over row in matrix!", 1);
  
  SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 
		       0, // xShift
		       -1, // yShift
		       1, // Dimensions
		       2, // Compute diff
		       LevelNum); // Compute only ResidualVec
  checkForCudaErrors("Sum over Residual vector!", 1);
  
  //  std::cout << "SumDiff: " << SumDiff << "\n" << std::endl;
  #ifdef __GPU__
  cudaMemcpy( &SumDiff, d_ResTaueNorm, sizeof(ValueType), cudaMemcpyDeviceToHost);
  checkForCudaErrors("MemCpy of SumDiff from GPU!", 1);
  #else
  SumDiff = (*d_ResTaueNorm);
  #endif

    // Compute L2Norm
#ifdef __GPU__
    ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum-1, 1, 0);
#else
    memset (d_ResidualVec, 0.0, (YNum)*sizeof(ValueType));
#endif
    
    checkForCudaErrors("After second level reset function", 1);
    
    //    SumRows DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 0, LevelNum);
    SumRowsBlockwise DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 0, LevelNum, color);
    checkForCudaErrors("Sum over row in matrix!", 1);
    SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 0/*xShift*/, -1, 1, 2, LevelNum); // Compute only ResidualVec 
    // (i.e. sum over 0 row in ResidualVec)
    checkForCudaErrors("Sum over Residual vector!", 1);
    
    
#ifdef __GPU__
    cudaMemcpy( &SumTot, d_ResTaueNorm, sizeof(ValueType), cudaMemcpyDeviceToHost);
    checkForCudaErrors("MemCpy of SumTot from GPU!", 1);
#else
    SumTot = (*d_ResTaueNorm);
#endif

    std::cout << "SumTot: " << SumTot << " SumDiff "<< SumDiff << " color " << color  << std::endl;
  
    ResResult=(SumDiff)/(SumTot+10e-16);
  
  if (ResResult != ResResult) {
    dumpGPUArrayToFile(M, "M.data", LevelNum);
    dumpGPUArrayToFile(MNew, "MNew.data", LevelNum);
    std::cout << "The residual is NaN!" << std::endl;
    std::cout << "Residual: (SumDiff)/(SumTot+10e-16) = " << SumDiff << "/" << (SumTot+10e-16) << " = " << ResResult << std::endl;
    ResResult = 0.0;
    exit(1);
  };

  return ResResult;
};

/**
   This function smoothes the stress and velocity grids. On each level a series of iterations
   are computed. The number of iterations can varry depending on level.
   Compared with the iSOSIA::Iterate function this smooths block wise and not points wise. This
   process is more computational demanding because 4 velocities and 3 stresses must be updated in each
   iteration. The cost is a more effective smoother on corse grids.

   The cross strain-rate Exy is unfortunally not able to decouple in a red-black way. Therefore a 
   diagonal (zebra or 3-color) indexing is used with a decoupling between a strain-rate and velocity-stress interation.
 **/
void iSOSIA::IterateBlocks(const int ForceStressIteration, profile & prof, const int Cycles, double Converg) {

  int color = 0;
  int iter = 0;
  for (iter = 0; iter<Cycles; iter++) {

    // This needs to happen when using MG
    // because velocities have been updated with prolong
    //    if (LevelNum == 0)
    BlockComputeSurfaceVelocity(color, 1, 0);
    BlockComputeSurfaceVelocity(color, 1, 1);

    ComputeStrainRate();

    int run = 0;
    ResTaueNorm = 1e10;

    // Iterate one color
    while ( ResTaueNorm > Converg  && (LevelNum < 1 || ForceStressIteration)) {

      BlockIterateStress(color, 0);
      checkForCudaErrors("After stress iteration", 1); 

      ResTaueNorm = BlockComputeResidual(d_Taue, d_TaueSuggested, 0, 0, color);            
      std::cout << "Stress residual on level " << LevelNum << " " << ResTaueNorm << "\n";

      if (SmootherIterations < run ) {
	std::cout << "Solver did not converge stress iteration with " << run << " iterations and residual " << ResTaueNorm  << "\n";
	break;
      };

      // Update i,j
      BlockIterateVelocity(0, 0, color, 0);
      // Update remaining block (i+1,j and i,j+1)
      BlockIterateVelocity(0, 0, color, 1);

      /* NOTE: Sliding temperally removed during testing
      ComputeSlidingVelocity();
      // Update remaining block (i+1,j and i,j+1)
      // Note: Perhaps this update will not work in block mode?
      ComputeSlidingVelocity(1);
      */

      // Update velocity with approx. of surface VelocityResidua
      BlockComputeSurfaceVelocity(color, 1, 0);
      BlockComputeSurfaceVelocity(color, 1, 1);

      SetBoundaries();
      checkForCudaErrors("After velocity iteration", 1);     
    }
    
    // Switch color
    if (color == 1) {
      color = 0;
    } else {
      color = 1;
    };

    run++;

  }
}

/**
   This function smoothes the stress and velocity grids. On each level a series of iterations
   are computed. The number of iterations can varry depending on level.
 **/
void iSOSIA::Iterate(const int ForceStressIteration,
		     const int ForceVelocityIteration,
		     profile & prof,
		     const int Cycles,
		     double Converg) {

  int iter = 0;
  for (iter = 0; iter<Cycles; iter++) {

    // This needs to happen when using MG
    // because velocities have been updated with prolong
    //    if (LevelNum == 0)
    ComputeSurfaceVelocity(1);

    ComputeStrainRate();

    int run = 0;
    ResTaueNorm = 1e10;

    while ( ResTaueNorm > Converg  && (LevelNum < 1 || ForceStressIteration)) {

      IterateStress(1, 0);
      checkForCudaErrors("After stress iteration", 1); 

      ResTaueNorm = ComputeResidual(d_Taue, d_TaueSuggested, 0, 0);            
      //      std::cout << "Stress residual on level " << LevelNum << " " << ResTaueNorm << "\n";
      if (SmootherIterations <= run ) {
	//	std::cout << "Solver did not converge stress iteration with " << run << " iterations and residual " << ResTaueNorm  << "\n";
	break;
      };

      run++;
    }
    glob_StressIterations += run;

    if (ForceVelocityIteration) {
    if (iter == 0 && LevelNum == 0) {
      // We have most likely just had a 
      // prolong. Then do a largere update
      IterateVelocity(0, 1);
    } else {
      IterateVelocity(0, 0);
    }

    ComputeSlidingVelocity();
   
    // Update velocity with approx. of surface VelocityResidua
    ComputeSurfaceVelocity(1);
    }
    SetBoundaries();
    checkForCudaErrors("After velocity iteration", 1);     

 };

};

/**
 * Computes if the current residual should be updated
 * 
*/
void iSOSIA::ResidualCheck(const int CurrentIter,            // Current Iteration
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
	
  if (LastIter > 0) {
    // Compute next residual check
    
    // Compute gradient. Convergence is linear in log space
    double grad = ((log(CurrentResidual/LastResidual)))/(CurrentIter - LastIter); // grad = (y2 - y1)/(x2 - x1)

    if (grad > 0.0 && LastResidual > 0.1) { 
      std::cout << "Residual is going up! Residual is " << CurrentResidual << std::endl; 
      //exit(EXIT_FAILURE);
    };
    
    // Find cycle where code wil converge if rate continues
    ExpectedConvergence = CurrentIter+log(ConvergenceRule/1.0)/grad; // x = (log(y) - log(b))/a
    
    int DistanceToConverge = ExpectedConvergence - CurrentIter;
    
    if (DistanceToConverge < SingleStp
	|| CurrentResidual < 0.01) {
      // Check again in next iteration
      NextIter = CurrentIter + 1;
    } else {
      // Check residual at next iter
      NextIter = CurrentIter + (int) DistanceToConverge*StpLen;
      
      if (NextIter > ExpectedConvergence) {
	NextIter = ExpectedConvergence;
      };
    };
  } else {
    // First run
    // Try 10 Cycles and check again
    NextIter = CurrentIter+StartIteration;
  };
  
  LastResidual = CurrentResidual; // Save the result
  LastIter = CurrentIter;    // Save iteration
 
};

/**
 * Call GPU function to set boundary conditions
 * 
 * SetVariable can take the following values,
 *  0: Set BC for Vx, Vy, Sxx, Syy, Sxy
 *  1: Set BC for Exym
 *
 * @param SetVariable The variable you want to set BC's on. 
 */
void iSOSIA::SetBoundaries(const int SetVariable) {

  /*
  if (SetVariable == 5) {
    // Special case to set boundaires during prolong routine
    setBoundaries DEVICESPECS (d_VxRes, d_VyRes, d_Vxs, d_Vys, d_Sxx, d_Syy, d_Sxy, d_Sxxm, d_Syym, d_Sxym, d_Exy, d_Exyc, SetVariable, LevelNum);
  } else if (SetVariable == 4) {
    setBoundaries DEVICESPECS (d_Vy, d_Vy, d_Vxs, d_Vys, d_dbdx_c, d_dhdx_c, d_dhdy_c, d_dbdy_c, d_Syym, d_Sxym, d_Exy, d_Exyc, 0, LevelNum);
  } else  {
    setBoundaries DEVICESPECS (d_Vx, d_Vy, d_Vxs, d_Vys, d_Sxx, d_Syy, d_Sxy, d_Sxxm, d_Syym, d_Sxym, d_Exy, d_Exyc, SetVariable, LevelNum);
  };
  */
  
  checkForCudaErrors("Post boundary call!", 1);
  
};


ValueType iSOSIA::VelocityResidual(profile &prof) {

  // Loop over stress iterations
  /*
  Time tbegin(boost::posix_time::microsec_clock::local_time());
  TimeDuration dt;  
  ValueType MaxNormX = 0.0;
  ValueType MaxNormY = 0.0;
  ValueType LToNormX = 0.0;
  ValueType LToNormY = 0.0;
  */
  ResVxNorm = ComputeResidual(d_Vxd, d_VxSuggested, 1, 0);
  // LToNormX = ComputeL2Norm(d_VxRes, d_VxRes, 1, 0);
  //MaxNormX = ComputeMaxNorm(d_VxRes);
  checkForCudaErrors("After vx residual computation", 1);
  ResVyNorm = ComputeResidual(d_Vyd, d_VySuggested, 0, 1);
  //LToNormY = ComputeL2Norm(d_VyRes, d_VyRes, 0, 1);
  //MaxNormY = ComputeMaxNorm(d_VyRes);
  checkForCudaErrors("After vy residual computation", 1);
  /*
  Time tend(boost::posix_time::microsec_clock::local_time());
  dt = tend - tbegin;
  prof.tResidual += (double)(dt.total_microseconds())/1000000; // Decial seconds
  prof.nResidual += 1;      
  */
  if (ResVxNorm < 1e-16 && ResVyNorm < 1e-16 && 0) {
    std::cout << "Residual is now at machine precision" << std::endl;
    return (1e-16);
  };


  if ((ResVxNorm == 0.0 || ResVyNorm == 0.0) && 0) {
    std::cout << "Velocity residual is 0.0. This should not be posible." << std::endl;
    exit(EXIT_FAILURE);
  };

  // Return greatest residual
  if (ResVxNorm > ResVyNorm) {
    //    std::cout << "ResVxNorm " << ResVxNorm <<  "L2NormX " << LToNormX << " MaxNorm "<< MaxNormX << std::endl;
    return ResVxNorm;
  } else {
    //    std::cout << "ResVyNorm " << ResVyNorm << " L2NormY " << LToNormY  << " MaxNorm " << MaxNormY << std::endl;
    return ResVyNorm;
  };
};


/**
 * Computes a mean of the sliding velocity
 **/
ValueType iSOSIA::SlidingResidual() {
  // Compute simple mean of sliding residual
  // The residual is defined as (Vb - Sliding)^2/Vb^2
  ValueType VxbNorm = ComputeResidual(d_Vxb, d_VxbSuggested, 0, 0);
  ValueType VybNorm = ComputeResidual(d_Vyb, d_VybSuggested, 0, 0);
  
  if (VxbNorm > VybNorm) {
    return VxbNorm;
  } else {
    return VybNorm;
  }

};


ValueType iSOSIA::HydrologyResidual_h() {
  // Compute simple mean of hydrology residual
  return ComputeResidual(d_hw, d_hwSuggested, 0, 0);
};

ValueType iSOSIA::HydrologyResidual_s() {
  // Compute simple mean of hydrology residual
  return ComputeResidual(d_Pw, d_PwSuggested, 0, 0);
  /*
  ValueType ScxRes = ComputeResidual(d_Scx, d_ScxSuggested, 0, 0);
  ValueType ScyRes = ComputeResidual(d_Scy, d_ScySuggested, 0, 0);

  if (ScxRes > ScyRes) {
    return ScxRes;
  } else {
    return ScyRes;
  }
  */
};

void iSOSIA::updateWantedOnLevel() {

  
  IterateStress(1, 1);
  checkForCudaErrors("After stress iteration", 1); 
  
  SetBoundaries();    
  IterateVelocity(1, 0);
  checkForCudaErrors("After velocity iteration", 1); 
  
};

ValueType iSOSIA::ComputeL2Norm(ValueType *M, ValueType *MNew,
				  const unsigned int xShift, const unsigned int yShift)  {
  // Note MNew is only a dummy variable for compatability!
  ValueType SumDiff = 0.0;
  ValueType SumTot = 0.0;

  #ifdef __GPU__
    ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum-1, 1, 0);
  #else
    memset (d_ResidualVec, 0.0, (YNum)*sizeof(ValueType));
  #endif
    
  checkForCudaErrors("After level reset function", 1);

  // Sum the square of all rows - 0 indicates that all values should be squared
  SumRows DEVICESPECS (M, MNew, d_ResidualVec, xShift, yShift, YNum-1, 0, LevelNum);
  checkForCudaErrors("Sum over row in matrix!", 1);
  
  // Sum the squared together
  SumRows DEVICESPECS (M, d_ResidualVec, d_ResTaueNorm, 
		       0, // xShift
		       -1, // yShift
		       1, // Dimensions
		       2, // Compute diff
		       LevelNum); // Compute only ResidualVec
  checkForCudaErrors("Sum over Residual vector!", 1);

  //  std::cout << "SumDiff: " << SumDiff << "\n" << std::endl;
  #ifdef __GPU__
  cudaMemcpy( &SumDiff, d_ResTaueNorm, sizeof(ValueType), cudaMemcpyDeviceToHost);
  checkForCudaErrors("MemCpy of SumDiff from GPU!", 1);
  #else
  SumDiff = (*d_ResTaueNorm);
  #endif


  // |x| = sqrt(x_1^2 + x_2^2 + x_3^2 + ... + x_n^2)
  double ResResult=sqrt(SumDiff);
  
  if (ResResult != ResResult) {
    std::cout << "The residual is NaN!" << std::endl;
    std::cout << "Residual: (SumDiff)/(SumTot+10e-16) = " << SumDiff << "/" << (SumTot+10e-16) << " = " << ResResult << std::endl;
    exit(1);
  };

  return ResResult;
};


void iSOSIA::ResetDeviceGrid(const int mode ) {

  // Mode = 1 is default and will reset all grids

  if (mode == 0 || mode == 1) {
    ResetLevel DEVICESPECS ( d_Vx     , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Vy     , 0.0, LevelNum);
  };


  if (mode == 2 || mode == 1) {
    ResetLevel DEVICESPECS ( d_VxRes  , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_VyRes  , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_TaueRes, 0.0, LevelNum);
  };

  if (mode == 1) {
    ResetLevel DEVICESPECS ( d_Taue   , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_VxR    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_VyR    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_TaueR  , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Sxx    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Syy    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Sxy    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Sxxm   , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Syym   , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Sxym   , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Exx    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Eyy    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Exy    , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Exyc   , 0.0, LevelNum);
    ResetLevel DEVICESPECS ( d_Ezz    , 0.0, LevelNum);
  };
  /*
  ResetLevel DEVICESPECS ( d_Vx, 0.0, LevelNum);
  ResetLevel DEVICESPECS ( d_Vy, 0.0, LevelNum);
  ResetLevel DEVICESPECS ( d_Taue, 0.0, LevelNum);
  */
  checkForCudaErrors("Grid reset", 1);
};


void iSOSIA::FixVelocityGradients( const int mode ) {
  if (mode == 1) {
    // Fix vx
    cuFixVelocityGradients DEVICESPECS (d_Vxc, 1, YNum-2, 2, XNum-2,  LevelNum);
  } else if (mode == 2)  {
    // Fix vy
    cuFixVelocityGradients DEVICESPECS (d_Vy, 2, YNum-2, 1, XNum-2,  LevelNum);
  } else if (mode == 3)  {
    // Fix vy
    cuFixVelocityGradients DEVICESPECS (d_dbdx_c, 1, YNum-2, 1, XNum-2,  LevelNum);
    cuFixVelocityGradients DEVICESPECS (d_dbdy_c, 1, YNum-2, 1, XNum-2,  LevelNum);
    cuFixVelocityGradients DEVICESPECS (d_dhdx_c, 1, YNum-2, 1, XNum-2,  LevelNum);
    cuFixVelocityGradients DEVICESPECS (d_dhdy_c, 1, YNum-2, 1, XNum-2,  LevelNum);
  };
};


void iSOSIA::dumpGPUArrayToFile(const ValueType *Array, const char FILENAME[30], const int level) {
  char file[200];
  FILE *fp;
  printf("Writing output file\n");
  /*create output file*/
  sprintf(file,"output_%s.dump",FILENAME);
  if ((fp = fopen(file,"w")) == NULL) {
    printf("could not open file for writing\n");
    exit(1);
  }
  else {

    ValueType *tmpArr;
    tmpArr  = new ValueType[GridSize];
    // Copy array from GPU to host
    #ifdef __GPU__
    cudaMemcpy( tmpArr,  Array,  GridSize, cudaMemcpyDeviceToHost ); 
    #endif

    for (int i = 0; i<XNum; i++) {
      for (int j = 0; j<YNum; j++) {
	fprintf(fp," %3.4e ", tmpArr[j*XNum+i]);
      }
      fprintf(fp,"\n");
    }  
    delete[](tmpArr);
    fclose(fp);
  }
};

void iSOSIA::MaxError(ValueType & XError, ValueType & YError, ValueType & XYError ) {
  XError = 1e30;
  YError = 1e30;
  XYError = 1e30;

  for (int i = 1; i<XNum; i++) {
    for (int j = 1; j < YNum; j++) {
      if ( Sxx[i*XNum+j] < XError && Sxx[i*XNum+j] != 0.0)
	XError = Sxx[i*XNum+j];

      if ( Syy[i*XNum+j] < YError && Syy[i*XNum+j] != 0.0)
	YError = Syy[i*XNum+j];

      if ( Sxy[i*XNum+j] < XYError && Sxy[i*XNum+j] != 0.0)
	XYError = Sxy[i*XNum+j];

    };
  };

};
 

  /**
   * Set the current residual variables on the device to a given value.
   * This function is needed to initilize variables before use.
   *
   * @param Value The value to set the variables to
   */
void iSOSIA::setResidual(ValueType Value) {
  ResetLevel DEVICESPECS (d_ResidualVec, 0.0, YNum, 0, 0);
  checkForCudaErrors("Reset of d_ResidualVec!", 1);
  setVariable DEVICESPECS (d_ResTaueNorm, Value);
  checkForCudaErrors("Reset of d_ResTaueNorm!", 1);
  setVariable DEVICESPECS (d_ResVxNorm, Value);
  checkForCudaErrors("Reset of d_ResVxNorm!", 1);
  setVariable DEVICESPECS (d_ResVyNorm, Value);
  checkForCudaErrors("Reset of d_ResVyNorm!", 1);
};

// Clones (or copies) the array g_in to g_out
void iSOSIA::cloneArray(ValueType *g_in, ValueType *g_out) {

  #ifdef __GPU__
  dim3 GPUGridSizeTopo = this->GPUGridSize;
  GPUGridSizeTopo.x += 1;
  GPUGridSizeTopo.y += 1;
  
  cuCloneArray <<<GPUGridSizeTopo, this->GPUBlockSize>>> (g_in, g_out, LevelNum);
  checkForCudaErrors("Array cloning", 1);
  #else
  cuCloneArray (g_in, g_out, LevelNum);
  #endif
}

#endif
