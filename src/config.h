/* ##############################################################
    Copyright (C) 2011 Christian Brædstrup

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
################################################################# */

//    Config file for CPU
//
//    Contact:
//      Christian Brædstrup
//      christian.fredborg@geo.au.dk
//      Ice, Water and Sediment Group
//      Institut for Geoscience
//      University of Aarhus
//


#ifndef _FILE_HANDLE_
#define _FILE_HANDLE_

#include <iostream>
#include <fstream>

#endif

#ifdef INT_2_BYTES
typedef char int8;
typedef int int16;
typedef long int32;
#else
typedef char int8;
typedef short int16;
typedef int int32;
#endif

#ifndef CONFIG_H_
#define CONFIG_H_

#define maxLevel 8    // Max number of levels
typedef double ValueType;

#ifdef __CPU__
//  #include "vector_functions.h"

struct uint3
{
    unsigned int x, y, z;
};

typedef uint3 dim3;

#endif

extern const char* program_name;

#include <vector>
#include <typeinfo> // Test type of value
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <getopt.h> // POSIX operating system API
#include <iomanip>
#include "constMemStruct.cuh"
#include "../../datatypes.h" // Datatypes from SPM

#ifndef __DATATYPES_H__
  #include "datatypes.h"
#endif

#ifdef __iSOSIA__
typedef iSOSIA LevelType;
#endif
#include "Error.h"
#include "Log.h"



#ifdef __GPU__
#define PLATFORM __global__
#define DEVICESPECS <<<GPUGridSize, GPUBlockSize>>>
#include "cuPrintf.cuh"
#else
#define PLATFORM 
#define DEVICESPECS 
#include <math.h>
#include <omp.h>
#endif

// THRUST FUNCTIONS
// square<T> computes the square of a number f(x) -> x*x
#ifdef __GPU__
template <typename T>
struct square
{
    __host__ __device__
    T operator()(const T& x) const { 
      return x * x;
    }
};
#endif


// MISC. UTILITY FUNCTIONS

// Error handler for CUDA GPU calls. 
//   Returns error number, filename and line number containing the error to the terminal.
//   Please refer to CUDA_Toolkit_Reference_Manual.pdf, section 4.23.3.3 enum cudaError
//   for error discription. Error enumeration starts from 0.
void checkForCudaErrors(const char* checkpoint_description, bool DoSync);

#ifdef __GPU__
void checkForCudaErrors(const char* checkpoint_description);
__global__ void reduction(const ValueType *g_idata, ValueType *g_odata, const int level); // Only avaible to GPU
#endif

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
//#define ERR(e) {printf("Error: %s\n", nc_strerror(e)); return 2;}
void msg (FILE* stream, const char* str, int type, int exit_code);

//////////////////////
// Shared functions //
// prototypes       //
//////////////////////
/*
void restrictLevel(ValueType *d_Resx, ValueType *d_Resy, ValueType *d_Resc, 
		   ValueType *d_Rx, ValueType *d_Ry, ValueType *d_Rc, 
		   ValueType *d_Etanf, ValueType *d_Etanc, 
		   int level, dim3 grid_size, dim3 block_size, 
		   int XNum, int YNum, int verbose, int RestrictOperator);

void prolongLevel(ValueType *d_Vxf, ValueType *d_Vyf, ValueType *d_Pf, 
		  ValueType *d_Vxc, ValueType *d_Vyc, ValueType *d_Pc, 
		  ValueType *d_Etanf, ValueType *d_Etanc, 
		  int level, dim3 grid_sizef, dim3 block_sizef, dim3 grid_sizec, dim3 block_sizec, 
		  int XNum, int YNum, ValueType vkoef, ValueType pkoef, 
		  int verbose, int ProlongOperator);
*/
//void initializeMG(MultigridStruct *MGS);
void AllocateMGLevel(int LevelNum, MultigridLevel* Level, size_t gridSize);
void SetConstMem(ConstantMem* h_ConstMem);



PLATFORM void Restrict(const ValueType* FineGrid, ValueType* CoarseGrid, const int level, int UB, int LB, const int mode);
PLATFORM void Prolong( ValueType* FineGrid, const ValueType* CoarseGrid, ValueType Koef, int Nodes, int FineLevel, int UB, int LB );
PLATFORM void ResetLevel(ValueType *g_M, ValueType Value, int level);
PLATFORM void ResetLevel(ValueType *g_M, ValueType Value, const unsigned int XDim, const unsigned int YDim, int level);
PLATFORM void fillGhost( ValueType *g_M, int RB, int LB, int UB, int BB, int level, int mode);

PLATFORM void restriction( ValueType *M_f, ValueType *M_c, 
			   ValueType dx, ValueType dy, 
			   bool useEta, ValueType *etanf, ValueType *etanc, 
			   int level, int LB, int UB);

PLATFORM void prolongation( ValueType *M_f, ValueType *M_c, 
			    ValueType dx, ValueType dy, 
			    bool useEta, ValueType *etanf, ValueType *etanc, 
			    ValueType Koef, int level, int LB, int UB, int mode);

PLATFORM void setVariable(ValueType *g_Var, ValueType Value);

#ifdef __iSOSIA__

unsigned int nextPow2( unsigned int x );

/* Hydrology */
PLATFORM void g_hydrology_update_sy(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
				    ValueType *g_M,
				    const ValueType *g_Psi, const ValueType *g_Pw, const ValueType *g_Sc,			     
				    const ValueType *g_Qc, const ValueType *g_qs, const ValueType *g_tn,
				    const ValueType dt, 
				    const ValueType *g_IceTopo, const ValueType *g_BedTopo,
				    const int level, const int color, const bool UpdateWanted);

PLATFORM void g_hydrology_update_sx(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
				    ValueType *g_M,
				    const ValueType *g_Psi, const ValueType *g_Pw, const ValueType *g_Sc,			     
				    const ValueType *g_Qc, const ValueType *g_qs, const ValueType *g_tn,
				    const ValueType dt, 
				    const ValueType *g_IceTopo, const ValueType *g_BedTopo,
				    const int level, const int color, const bool UpdateWanted);

PLATFORM void g_hydrology_update_h(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, ValueType *g_old,
				   const ValueType *g_PwOld, const ValueType *g_Pw,
				   const ValueType *g_Qcx, const ValueType *g_qsx,
				   const ValueType *g_Qcy, const ValueType *g_qsy, 
				   const ValueType *g_Psi, const ValueType *g_mcx, const ValueType *g_mcy, 
				   const ValueType *g_R, const ValueType dt, const ValueType *g_Sc,
				   const ValueType *g_tb, const ValueType *g_vb,
				   const ValueType *g_IceTopo, const ValueType *g_BedTopo,
			  const int level, const int color, const bool UpdateWanted);

PLATFORM void g_hydrology_update_psi(ValueType *g_Psi, 
				     const ValueType *g_bed, 
				     const ValueType *g_Pw,
				     const int level);

PLATFORM void g_hydrology_update_flux(
				      ValueType *g_Qxc, ValueType *g_Qyc,
				      ValueType *g_qxs, ValueType *g_qys,
				      const ValueType *g_Psi, const ValueType *g_Sc, 
				      const ValueType *g_hw,
				      const ValueType *g_IceTopo, const ValueType *g_BedTopo,
				      const int level, const int color, const bool UpdateWanted
);

PLATFORM void g_hydrology_update_pressure(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, ValueType *g_old,
					  const ValueType *g_hw, const ValueType *g_tn,
					  const ValueType *g_tb, const ValueType *g_Vb,
					  const ValueType *g_qsx, const ValueType *g_qsy,
					  const ValueType *g_R, const double dt,
					  const ValueType *g_dbdx, const ValueType *g_dbdy,
					  const ValueType *g_Ice, const ValueType *g_Bed,
					  const int level, const int color, const bool UpdateWanted);

/*
PLATFORM void g_compute_coefficients(ValueType *g_Vx, ValueType *g_Vy,
				     ValueType *g_VxRes, ValueType *g_VyRes,
				     const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
				     ValueType *g_Taue, ValueType* g_TaueRes,
				     const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
				     const int level, const int color, const int update_velocity);
*/
PLATFORM void g_Compute_test_topography(ValueType *Ice, ValueType *Bed,
					ValueType *dbdx, ValueType *dbdy,
					ValueType *dhdx, ValueType *dhdy,
					ValueType *dbdx_w, ValueType *dbdy_w,
					ValueType *dhdx_w, ValueType *dhdy_w,
					int mode, int level);

PLATFORM void cuCloneArray(ValueType *g_in, ValueType *g_out, const int level);

PLATFORM void cuFixVelocityGradients(ValueType *g_M, const int UB, const int BB, const int LB, const int RB, const int level); //(ValueType *g_in, ValueType *g_out, const int level);
PLATFORM void g_update_effective_stress(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
					const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
					const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
					const ValueType *g_dhdx, const ValueType *g_dhdy,
					const int level, const int color, const bool UpdateWanted);

PLATFORM void g_update_vx(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			  ValueType *g_Vx, ValueType *g_Vxs, ValueType *g_Vxb, ValueType *g_Ezz,
			  const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
			  const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
			  const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
			  const ValueType *g_dhdx, const ValueType *g_dhdy,
			  const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted,
			  const bool PostProlongUpdate, const int Blockwise = 0);

PLATFORM void g_update_vy(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			  ValueType *g_Vy, ValueType *g_Vys, ValueType *g_Vyb, ValueType *g_Ezz,
			  const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
			  const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
			  const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
			  const ValueType *g_dhdx, const ValueType *g_dhdy,
			  const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted,
			  const bool PostProlongUpdate, const int Blockwise = 0);

// Basal sliding update terms
PLATFORM void g_update_vxb(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			   ValueType *g_Vxd, ValueType *g_Vx,
			   const ValueType *g_Tbx,
			   const ValueType *g_Ts, const ValueType *g_Vb, 
			   const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
			   const int level, const int color, const int Blockwise = 0);

PLATFORM void g_update_vyb(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			   ValueType *g_Vyd, ValueType *g_Vy,
			   const ValueType *g_Tby,
			   const ValueType *g_Ts, const ValueType *g_Vb, 
			   const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
			   const int level, const int color, const int Blockwise = 0);

PLATFORM void g_Compute_sliding_residual(ValueType *g_VbRes, const ValueType *g_Vbx, const ValueType *g_Vby, const ValueType *g_Vb, const int level);   

PLATFORM void g_compute_sliding_terms(ValueType *g_Tbx, ValueType *g_Tby, ValueType *g_Tbz, ValueType *g_Vb, ValueType *g_Ts, ValueType *g_Tn, ValueType *g_beta,
				      const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy,
				      const ValueType *g_Pw, const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
				      const ValueType *g_dhdx, const ValueType *g_dhdy, const ValueType *g_dbdx, const ValueType *g_dbdy,
				      const int level, const int color);


PLATFORM void cuGetGradients(const ValueType* g_icesurf, 
		      const ValueType *g_bedsurf, 
		      ValueType *g_dhdx, ValueType *g_dhdy, 
		      ValueType *g_dbdx, ValueType *g_dbdy, 
		      int level);

PLATFORM void cuInterpGradients(const ValueType *g_dhdx,   const ValueType *g_dhdy, 
				const ValueType *g_dbdx,   const ValueType *g_dbdy,
				ValueType *g_dhdx_c, ValueType *g_dhdy_c, 
				ValueType *g_dbdx_c, ValueType *g_dbdy_c, 
				int level);

PLATFORM void SumRows(const ValueType *g_M, const ValueType *MNew, ValueType *g_Residual, int xShift, int yShift, unsigned int dimensions, const int ComputeDiff, int level);
PLATFORM void SumRowsBlockwise(const ValueType *g_M, const ValueType *MNew, ValueType *g_Residual, int xShift, int yShift, unsigned int dimensions, const int ComputeDiff, int level, int color);
PLATFORM void MaxRows(const ValueType *g_M, ValueType *g_MaxInRow, int level);
PLATFORM void ComputeL2Norm( const ValueType *M, const ValueType *ResM, ValueType *Norm, int xShift, int yShift, int level);
PLATFORM void ComputeError(const ValueType* u, const ValueType *v, ValueType *error_out, const int level, const int xShift, const int yShift);

PLATFORM void g_Compute_strain_rate(ValueType *g_Vx,  ValueType *g_Vy,
				    ValueType *g_Vxs, ValueType *g_Vys,
				    ValueType *g_Vxb, ValueType *g_Vyb,
				    ValueType *g_Exx, ValueType *g_Eyy, 
				    ValueType *g_Exy,
				    ValueType *g_Ezz,
				    ValueType *g_IceTopo, ValueType* g_BedTopo,
				    const ValueType *g_dhdx, const ValueType *g_dhdy,
				    const ValueType *g_dbdx, const ValueType *g_dbdy,
				    int level, int color);

PLATFORM  void g_Compute_Exy_cross_terms (const ValueType *g_Vx, const ValueType *g_Vy,
					  ValueType *g_Exy,
					  const int level);
  
PLATFORM void VerifyConstMem(ConstantMem* d_ConstMem, int* Error);

PLATFORM void setBoundaries( ValueType *g_Vx, ValueType *g_Vy, ValueType *g_Vxs, ValueType *g_Vys, 
			     ValueType *g_Sxx, ValueType* g_Syy, ValueType* g_Sxy,
			     ValueType *g_Sxxm, ValueType* g_Syym, ValueType* g_Sxym,
			     ValueType* g_Exy, ValueType* g_Exyc,
			     const int SetValue, int level);

PLATFORM void g_update_stress(ValueType *g_Sxx, ValueType *g_Syy, ValueType *g_Sxy,
			      const ValueType *g_Exx, const ValueType *g_Eyy, const ValueType *g_Exy, 
			      const ValueType *g_Taue, const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
			      const int level, const int color);


PLATFORM void g_Interp_Exy(const ValueType *g_Exy, ValueType *g_Exym, int level);

PLATFORM void g_CenterToCornerInterp(ValueType* g_In, ValueType* g_Out, const int level);

#endif


#endif



#ifdef __CPU__
ValueType grad( const int Row, const int Col, //! Row and column index
      const ValueType *M, //! Array to get values from
      const int xDisplacement, //! Compute gradient along x axis
      const int yDisplacement, //! Compute gradient along y axis
      const int level //! Current multigrid level
      );
ValueType gradVelocity( const int Row, const int Col, //! Row and column index
			const ValueType *M, //! Array to get values from
			const int xDisplacement, //! Compute gradient along x axis
			const int yDisplacement, //! Compute gradient along y axis
			const int level //! Current multigrid level
			);
#endif
