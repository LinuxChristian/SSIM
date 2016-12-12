
#define __DATATYPES_H__
#include "config.h"

////////////////
// Structures //
////////////////

typedef struct MultigridStructTag {
  MultigridStructTag()
  : RequestedLevels(0), totIterNum(0), maxIterNum(0), Direction(0), startLevel(0), startIteration(0), IOUpdate(0), RestrictOperator(0), ProlongOperator(0), Cycle(0), NumberOfCycles(0), equations(0), ncid(0), CPU(0) {

    threadsPerBlock[0] = 0;
    threadsPerBlock[1] = 0;
    threadsPerBlock[2] = 0;

    numBlock[0] = 0;
    numBlock[1] = 0;
    numBlock[2] = 0;

    GPUGridSize.x = 0;
    GPUGridSize.y = 0;
    GPUGridSize.z = 0;

    GPUBlockSize.x = 0;
    GPUBlockSize.y = 0;
    GPUBlockSize.z = 0;
    
    TotalMemSize = 0;
    TotalConstMemSize = 0;
  };

  // Setup Multigrid solver
  int RequestedLevels; // Number of multigrid levels requested
  int totIterNum; // Total number of iterations
  int maxIterNum; // Max number of iterations
  int Direction; // DEFAULT: Restriction
  int startLevel; // DEFAULT: 0 (ONLY used for debugging!)
  int startIteration; // Used if file is reused
  int IOUpdate;
  int RestrictOperator;
  int ProlongOperator;
  int IterationsOnLevel[maxLevel]; // Number of GS iterations at each level
  int Cycle;  /* Number of current run cycles */

  // Random bits
  int NumberOfCycles;
  int equations; // Which equations to solve
  int* ncid;
  int CPU;
  
  // Setup CPU and CPU
  int threadsPerBlock[3];
  int numBlock[3]; // 
  dim3 GPUGridSize, GPUBlockSize;

  size_t TotalMemSize;
  size_t TotalConstMemSize;

  // Settings for Convergence Computation
  int FirstStp;               // Number of iterations to run before computing first residual
  double StpLen;                    // Lenght of step before next update
  int SingleStp;                    // When to compute single steps
  double Converg_s;
  double Converg_v;
} MultigridStruct;

typedef struct profileTag{
  profileTag()
  {
    tCycle      = 0.0; // Multigrid cycles
    nCycle      = 0;
    tGSIter     = 0.0; // Gauss-Seidel Iterations
    nGSIter     = 0;
    tStressIter = 0.0; // Stress Iterations
    nStressIter = 0;
    tVelIter    = 0.0; // Velocity Iterations
    nVelIter    = 0;
    tResidual   = 0.0; // Resudal computations
    nResidual   = 0;
    tProlong    = 0.0; // Prolong
    nProlong    = 0;
    tRestrict   = 0.0; // Restrict
    nRestrict   = 0;

    TotalRunTime= 0.0;
  }

  double tCycle; // Total time to run cycle
  int    nCycle; // Number of times V or F cycle was called
  double tGSIter;
  int    nGSIter;
  double tStressIter; // Total time running stress iteration 
  int    nStressIter; // Number of calls to stress iteration 
  double tVelIter;    // Total time running velocity iteration
  int    nVelIter;    // Number of calls to velocity iteration
  double tResidual;   // Total time running residual computation
  int    nResidual;   // Times function was called
  double tProlong; // Prolong
  int    nProlong;
  double tRestrict; // Restrict
  int    nRestrict;

  double TotalRunTime; // Total run time
} profile;


/*! \breif This is the base abstract class for different multigridlevels
 *
 * This class defines all functions and variables that the children classes need.
 * The class is in a has-a relationship with the multigrid class. In this way each Multigrid
 * instance has a multigrid level (often several). The multigrid levels get it's functions from the
 * children classes: Stokes_2d and iSOSIA. The MultigridLevel class is therefore only a abstract class
 * and cannot be declared as a instance.
 */
class MultigridLevel {

 public:
  friend class Multigrid;

  // Global Functions
  int GetLevel();
  void SetGPUGridSize(int x, int y, int z);
  void SetGPUBlockSize(int x, int y, int z);
  void SetFileOffset(unsigned int x, unsigned int y, unsigned int z);
  void SetDataSize(unsigned int x, unsigned int y, unsigned int z);
  void Memset(ValueType * Grid, double i);
  void Print(ValueType * Grid);

  /**
   // Pure virtual fuction to iterate over a given set of equations.
   // The kind of iteration is specified by the __2DStokes__ or __iSOSIA__ compiler flags.
   //    @param ncid NetCDF file ID
  */
  virtual void Iterate(const int ForceStressIteration,
		       const int ForceVelocityIteration, profile & prof, const int CycleNum,
		       double Converg) = 0;

  virtual void IterateBlocks(const int ForceStressIteration,
			     profile & prof, const int CycleNum,
		       double Converg) = 0;

  /**
   // Pure virtual fuction to copy the current level to CUDA device memory
   */
  virtual void CopyLevelToDevice() = 0;

  /**
   // Pure virtual fuction to copy the current level from CUDA device memory
   */
  virtual void CopyLevelToHost(const bool FileIO) = 0;

  /**
   // Pure virtual fuction to set the current level in device memory to zero.
   */
  virtual void ResetDeviceGrid(const int mode =1 ) = 0;

  /**
   // Pure virtual fuction to set boundaries around the current domain
   */
  virtual void SetBoundaries(const int SetVariable = 0) = 0;
  
  /**
   // Pure virtual fuction to compute L2 Norm of velocities.
   */
  virtual ValueType VelocityResidual(profile &prof) = 0;

  /**
   // Pure virtual fuction to compute max error 
   */
  virtual void MaxError(ValueType & XError, ValueType & YError, ValueType & PError ) = 0; // TODO: Change input params

 protected:
  // Member functions and objects only available to inheriated classes

  //! Constructore
  MultigridLevel(const unsigned int LevelNum, 
		 const unsigned int XNum, const unsigned int YNum, 
		 const ValueType Dx, const ValueType Dy, 
		 const unsigned int IterationLevel,
		 const unsigned int startIteration);
  //! Destructor
  ~MultigridLevel();

  void SetDimensions(int XNum, int YNum); //!< Set dimensions of grid
  int LevelNum, //!< Number of current level
    XNum,       //!< Number of gridpoints in x-direction
    YNum;       //!< Number of gridpoints in y-direction
  ValueType dx, dy, pnorm;
  size_t GridSize;
  int SmootherIterations;
  
  dim3 GPUGridSize, GPUBlockSize;
};

#ifdef __iSOSIA__

/*!
 Class to hold iSOSIA class type
 
 The class is inherited from the MultigridLevel class and
 holds special functions to handle iSOSIA equations

 \verbatim embed:rst
 .. note::
    This class is only avaible if __iSOSIA__ is set in the makefile
 \endverbatim
 */

class iSOSIA : public MultigridLevel {
  
 public:
  friend class Multigrid;

  // Construct and destruct
  iSOSIA(const unsigned int LevelNum,
	 const unsigned int xNum, const unsigned int yNum, 
	 const ValueType Dx, const ValueType Dy, 
	 const unsigned int IterationLevel,
	 const unsigned int startIteration);// /*, const MultigridStruct & MGS, ConstantMem & ConstMem*/);
  ~iSOSIA();
  void RestrictTopography(iSOSIA* Fine, iSOSIA* Coarse);
  void VerifyInputData(int ncid, int NCVar, int ndims);

  // Pure virtual classes
  /* SPM function prototypes */
  //  void ReadLevelFromSPM(celltype** cells,hptype** hp,vptype** vp,cornertype** cp, meshtype &mesh, iproptype &iprop, int NumberOfLevels);
  //  void ReadLevelToSPM(celltype** cells,hptype** hp,vptype** vp,cornertype** cp, meshtype &mesh, iproptype &iprop, int NumberOfLevels);
  //  void ReadLevelToSPM(celltype (**cells),hptype ***hp,vptype **vp, cornertype **cp, meshtype &mesh, iproptype &iprop, int NumberOfLevels);
  //  void ReadLevelToSPM(hptype &hp, int NumberOfLevels);
  void FixSPMPointers(ValueType *SPMVx, ValueType *SPMVy, ValueType *SPMVxs, ValueType *SPMVys, 
		      ValueType *SPMVxb, ValueType *SPMVyb,
		      ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue, 
		      ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
		      ValueType *SPMIceTopo, ValueType *SPMBedTopo, ValueType *SPMPw,
		      ValueType *SPMTbx, ValueType *SPMTby, ValueType *SPMTbz, ValueType *SPMTs, ValueType *SPMTn, ValueType *SPMVb, ValueType *SPMbeta,
		      ValueType *SPMdhdx, ValueType *SPMdhdy, ValueType *SPMdbdx, ValueType *SPMdbdy,
		      ValueType *SPMScx, ValueType *SPMScy, ValueType *SPMhw, ValueType *SPMqsx, ValueType *SPMqsy,
		      ValueType *SPMQcx, ValueType *SPMQcy, ValueType *SPMR, ValueType *SPMPsi, ValueType *SPMMcx, ValueType *SPMMcy);

  void setTestATopo(ValueType XStp, ValueType YStp, ValueType XSize, ValueType YSize);

  void Iterate(const int ForceStressIteration,
	       const int ForceVelocityIteration, profile & prof, 
	       const int CycleNum, double Converg);
  void IterateBlocks(const int ForceStressIteration, profile & prof, 
		     const int Cycles, double Converg);
  void CopyLevelToDevice();
  void CopyLevelToHost(const bool FileIO);
  void ResetDeviceGrid(const int mode = 1);
  void SetBoundaries(const int SetVariable = 0);
  void FillGhost(ValueType *M, const int XStart, const int YStart, const int mode = 1);
  void MaxError(ValueType & XError, ValueType & YError, ValueType & PError );
  ValueType VelocityResidual(profile &prof);
  ValueType SlidingResidual();
  void IterateStress(const int runs, const int updateWanted);
  void BlockIterateStress(const int color, const int updateWanted);
  void IterateVelocity(const int updateWanted, const bool PostProlongUpdate, const int Blockwise = 0);
  void BlockIterateVelocity(const int updateWanted, const bool PostProlongUpdate, const int color, const int Blockwise = 0);
  ValueType ComputeResidual(ValueType *M, ValueType *MDiff, const unsigned int xShift, const unsigned int yShift, const int normType);
  ValueType BlockComputeResidual(ValueType *M, ValueType *MDiff, const unsigned int xShift, const unsigned int yShift, const int color);
  void dumpGPUArrayToFile(const ValueType *Array, const char FILENAME[30], const int level);
  void ComputeSlidingVelocity(const int Blockwise = 0);
  void setResidual(ValueType Value);
  void InterpIceTopographyToCorner();
  void ResidualCheck(const int CurrentIter, const ValueType CurrentResidual, int &NextIter, int &LastIter, ValueType &LastResidual, int &ExpectedConvergence,
		     int StartIteration, double StpLen, int SingleStp, double ConvergenceRule);
  void updateWantedOnLevel();
  void ComputeErrorForProlong(const ValueType* u, const ValueType *v, ValueType *error, const int xShift, const int YShift);
  void ComputeStrainRate();
  void ComputeSurfaceVelocity(const int updateType);
  void BlockComputeSurfaceVelocity(const int color, const int updateType, const int Blockwise);
  void getGradients();
  void cloneArray(ValueType *g_in, ValueType *g_out);
  void FixVelocityGradients(const int mode);
  void solveHydrology(double dt);
  ValueType HydrologyResidual_h();
  ValueType HydrologyResidual_s();
  /**
   * Compute max error 
   * @param XSum Sum of XError
   */
  ValueType ComputeL2Norm(ValueType *M, ValueType *MDiff, const unsigned int xShift, const unsigned int yShift);
  ValueType ComputeMaxNorm(ValueType *d_Min);

 private:
  /* Host pointers */
  ValueType *IceTopo, //!< Ice Surface Topography
    *IceTopom, //!< Ice Surface Topography at base node
    *BedTopo; //!< Bed Topography
  ValueType *Sxx, *Syy, *Sxy;                // Stress Components
  ValueType *Sxxm, *Syym, *Sxym;                // Stress Components at base nodes (m for mean)
  ValueType *Vy, *VyRes, *VySuggested, *VyR;
  ValueType *Vxs;                               // X surface and basal/sliding velocities
  ValueType *Vxb, *VxbSuggested, *VxbR, *VxbRes; // X basal/sliding velocity
  ValueType *Vys;                               // Y surface velocities
  ValueType *Vyb, *VybSuggested, *VybR, *VybRes; // Y basal/sliding velocity
  ValueType *Vxd, *Vyd;                         
  ValueType *Vxc, *Vyc;                         // Values from coarse grid
  ValueType *Vx, *VxRes, *VxSuggested, *VxR;
  ValueType *Exx, *Eyy, *Exy, *Exyc, *Ezz;   // Strain Rate Components
  ValueType *Taue, *TaueRes, *TaueSuggested, *TaueR;
  ValueType ResTaueNorm, ResVxNorm, ResVyNorm; // Residuals
  ValueType *ResidualVec, *MaxResVec;
  ValueType *Tn; // Normal and effective pressure
  ValueType *dhdx, *dbdx, *dhdy, *dbdy;   // Gradients
  ValueType *dhdx_c, *dbdx_c, 
            *dhdy_c, *dbdy_c;   // Gradients in cell center
  ValueType *Tbx, *Tby, *Tbz, *Ts;        // Basal traction
  ValueType *Vb, *VbRes;  // Basal velocity
  int glob_StressIterations; // Global number of stress iterations
  
  // Hydrology  
  ValueType *Scx, *ScxRes, *ScxR, *ScxSuggested;
  ValueType *Scy, *ScyRes, *ScyR, *ScySuggested;
  ValueType *hw, *hwRes, *hwR, *hwSuggested, *hw_old;
  
  ValueType *Pw, *PwSuggested, *PwR, *PwRes, *Pw_old;

  ValueType *qsx, *qsy, *Qcx, *Qcy;
  ValueType *R, *Psi, *Mcx, *Mcy;
  ValueType *beta;

  /* Device pointers */
  ValueType *d_IceTopo, *d_IceTopom, *d_BedTopo;   
  ValueType *d_Sxx    , *d_Syy    , *d_Sxy;                   // Stress Components
  ValueType *d_Sxxm   , *d_Syym   , *d_Sxym;                  // Stress Components at base nodes (m for mean)
  ValueType *d_Exx    , *d_Eyy    , *d_Exy, *d_Exyc, *d_Ezz;  // Strain Rate Components (Note: Exy is a interpolated Strain Rate)
  ValueType *d_Vy, *d_VyRes, *d_VySuggested, *d_VyR;          // y-velocity components
  ValueType *d_Vx, *d_VxRes, *d_VxSuggested, *d_VxR;          // x-velocity components
  ValueType *d_Vxs;                                           // x-surface velocities
  ValueType *d_Vxb, *d_VxbSuggested, *d_VxbR, *d_VxbRes;       // x-basal/sliding velocity
  ValueType *d_Vys;                                           // y-surface velocities
  ValueType *d_Vyb, *d_VybSuggested, *d_VybR, *d_VybRes;       // x-basal/sliding velocity
  ValueType *d_Vxd, *d_Vyd;                         
  ValueType *d_Vxc, *d_Vyc;                                   // Values from coarse grid
  ValueType *d_Taue, *d_TaueRes, *d_TaueSuggested, *d_TaueR;  // Effective stress components
  ValueType *d_ResTaueNorm, *d_ResVxNorm, *d_ResVyNorm;
  ValueType *d_ResidualVec, *d_MaxResVec;
  ValueType *d_Tn; // Normal and effective pressure
  ValueType *d_dhdx, *d_dbdx, 
            *d_dhdy, *d_dbdy;   // Gradients
  ValueType *d_dhdx_c, *d_dbdx_c, 
            *d_dhdy_c, *d_dbdy_c;   // Gradients in cell center
  ValueType *d_Tbx, *d_Tby, *d_Tbz, *d_Ts;        // Basal traction
  ValueType *d_Vb, *d_VbRes;  // Basal velocity

  // Hydrology
  ValueType *d_Scx, *d_ScxRes, *d_ScxR, *d_ScxSuggested;
  ValueType *d_Scy, *d_ScyRes, *d_ScyR, *d_ScySuggested;
  ValueType *d_hw, *d_hwRes, *d_hwR, *d_hwSuggested, *d_hw_old;
  ValueType *d_qsx, *d_qsy, *d_Qcx, *d_Qcy;
  ValueType *d_R, *d_Psi, *d_Mcx, *d_Mcy;
  
  ValueType *d_Pw, *d_PwSuggested, *d_PwR, *d_PwRes, *d_Pw_old;
  
  ValueType *d_beta;

  /* Used to keep track of 
     calls to residual     */
  int StressNextResidualUpdate;
  int StressLastResidualUpdate;
  ValueType StressLastResidual;
  int StressExpectedConvergence;
};

// The type of equations to solve.
// Can be Stokes_2D, Stokes_3D and iSOSIA
// Note: Not all equations may be implemented
typedef iSOSIA LevelType;



#endif



/*!
 *
 */


#ifndef _Multigrid_
#define _Multigrid_

/* CLASS */
class Multigrid {
 public:
  friend class Log;
  Multigrid(const int CPU, const size_t TotalMemSize, const size_t TotalConstMemSize,
	    int SingleStp, double StpLen, int FirstStp, double Converg_s, double Converg_v);  
  //  void Initialize(MultigridStruct & MGS, ConstantMem & ConstMem);
  void CPU_Restrict(const MultigridLevel & FineLevel, MultigridLevel & CoarseLevel, const MultigridStruct & MGS );
  MultigridStruct GetMultigridSettings();

  void prolongLevel(ValueType *d_Vxc, ValueType *d_Vyc, ValueType *d_Pc, 
		    ValueType *d_Vxf, ValueType *d_Vyf, ValueType *d_Pf, 
		    ValueType *d_Etanc, ValueType *d_Etanf, 
		    int level, 
		    dim3 grid_sizef, dim3 block_sizef, dim3 grid_sizec, dim3 block_sizec, 
		    int XNum, int YNum, 
		    ValueType vkoef, ValueType pkoef, 
		    int verbose, int ProlongOperator,
		    LevelType *CurrentLevel, LevelType *NextLevel);

  void restrictLevel(ValueType *d_Resx, ValueType *d_Resy, ValueType *d_Resc, 
		     ValueType *d_Rx, ValueType *d_Ry, ValueType *d_Rc, 
		     ValueType *d_Etanf, ValueType *d_Etanc, 
		     int level, dim3 grid_size, dim3 block_size, 
		     int XNum, int YNum, int verbose, int RestrictOperator,
		     LevelType *CurrentLevel, LevelType *NextLevel);
    
  // Global helper functions
  void SetFirstLevel(int FirstLevel);
  void SetFirstLevel( );
  int GetFirstLevel( );
  int NumberOfCycles();
  void ComputeMaxError(int level);
  int GetIOUpdate();
  ValueType ComputeNorm(int level);
  unsigned int GetCurrentLevels();
  int GetNumberOfIterations();
  dim3 GetThreadsPerBlock();
  dim3 GetNumberOfBlocks();
  void SetCycleNum(const int num);
  int GetCycleNum();
  void UpdateCycleNum();
  void setResidual(const ValueType Res, const int CycleNum);
  ValueType getResidual(const int CycleNum);
  void ReduceRelaxation(const ValueType factor);
  void ComputeSurfaceVelocity(const int updateType);
  void ResidualCheck(const int CurrentIter, const ValueType CurrentResidual, 
		     int &NextIter, int &LastIter, ValueType &LastResidual, int &ExpectedConvergence,
		     int StartIteration, double StpLen, int SingleStp, double ConvergenceRule);
  void initializeGPU(size_t* TotalGlobalMem, size_t* TotalConstMem);
  void ComputeSurfaceGradients(const int Level);

  // Common level operations
  int GSCycle(profile & prof, bool doMG, const int ForceVelocity);
  int RunVCycle(profile &prof,bool ForceOneLevel, const int ForceVelocity);
  void RunWCycle(profile &prof);
  void RunFMG(profile &prof);
  void RestrictTopography( const int Fine, const int Coarse, const int CPU );
  ValueType VelocityResidual(profile &prof);
  ValueType SlidingResidual();
  void SolveHydrology(double dt);

  // Level specific functions
  void AddLevels(const int MaxStressIter);

  void FixSPMPointers(ValueType *SPMVx, ValueType *SPMVy, ValueType *SPMVxs, ValueType *SPMVys, 
		      ValueType *Vxb, ValueType *Vyb,
		      ValueType *Sxx, ValueType *Syy, ValueType *Sxy, ValueType *Szz, ValueType *Taue, 
		      ValueType *Exx, ValueType *Eyy, ValueType *Exy, ValueType *Ezz, 
		      ValueType *SPMIceTopo, ValueType *SPMBedTopo, ValueType *Pw,
		      ValueType *SPMTbx, ValueType *SPMTby, ValueType *SPMTbz, ValueType *SPMTs, ValueType *SPMTn, ValueType *SPMVb, ValueType *SPMbeta,
		      ValueType *SPMdhdx, ValueType *SPMdhdy, ValueType *SPMdbdx, ValueType *SPMdbdy,
		      ValueType *SPMScx, ValueType *SPMScy, ValueType *SPMhw, ValueType *SPMqsx, ValueType *SPMqsy,
		      ValueType *SPMQcx, ValueType *SPMQcy, ValueType *SPMR, ValueType *SPMPsi, ValueType *SPMMcx, ValueType *SPMMcy);
  void setTestATopo();
  void ComputeSliding();

  void CopyLevelToDevice(const int CopyLevel);
  void CopyLevelToHost(const int CopyLevel, const bool fileIO);
  void SetConstMemOnDevice();
  void DisplaySetup();
  void TransferSettingsToSSIM(int requestedLevels, int restrictOperator, int prolongOperator,
			      int *iterationsOnLevel, int equations, int startLevel, int cycles,
			      int xnum, int ynum, int L, int H, 
			      int *bc, double sRelax, double vRelax, double Gamma, double A,
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
			      ValueType L0, ValueType minbeta, ValueType C, 
			      ValueType Cs,
			      ValueType maxSliding, ValueType vbfac,
			      int dosliding, int slidingmode);
  void InterpIceTopographyToCorner(const int level);

 private:
  int CurrentLevelNum;  // Number of current created levels
  int Direction;
  int CustomError;
  int* d_CustomError;
  LevelType* CurrentLevel;
  LevelType* FirstLevel;
  LevelType* LastLevel;
  LevelType* Levels[maxLevel];
  ValueType* Residual;  // Pointer to a vector that holds the computes velocity residuals
  MultigridStruct MGS; // Allocate variables for MG
  ConstantMem ConstMem; // Constant memory structure on host
  ConstantMem* d_ConstMem; // Pointer to constant memory structure in global memory (for debugging!)
};

#endif
