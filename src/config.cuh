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

//    Config file for GPU
//
//    Contact:
//      Christian Brædstrup
//      christian.fredborg@geo.au.dk
//      Ice, Water and Sediment Group
//      Institut for Geoscience
//      University of Aarhus
//

#ifndef CONFIG_CUH_
#define CONFIG_CUH_

#define maxLevel 8    // Max number of levels
#define BLOCK_SIZE 32 // Size of shared memory to allocate
#define PADDING 1     // Thickness of ghost nodes

typedef double ValueType;

#ifdef __GPU__
  #include <cuda.h>
//  #include <cutil_math.h>
#endif

#include "constMemStruct.cuh"

////////////////////////
// Prototypes  DEVICE //
////////////////////////

#ifdef __GPU__
__device__ void RBGS( ValueType *Vx, ValueType *Vy, ValueType *P,
		      ValueType *Rx, ValueType *Ry, ValueType *Rc, // Righthand side
		      ValueType *Resx, ValueType *Resy, ValueType *Resc,
		      ValueType *Etan, ValueType *Etas,            // Viscosity
		      int level, int Row, int Col, int g_Row, int g_Col, int right);

__device__ ValueType vx_kernel( double *Vx, double *Vy, double *P, 
				double *R, ValueType* Res, 
				ValueType *Etan, ValueType *Etas, 
				int j, int i, int g_Row, int g_Col, int level, int right);

__device__ ValueType vy_kernel( double *Vx, double *Vy, double *P, 
				double *R, ValueType* Res, 
				ValueType *Etan, ValueType *Etas, 
				int j, int i, int g_Row, int g_Col, int level, int right);

__device__ ValueType p_kernel( double *Vx, double *Vy, double *P, 
			       double *R, ValueType* Res, 
			       ValueType *Etan, ValueType *Etas, 
			       int j, int i, int g_Row, int g_Col, int level, int right);

__device__ void computeRight(ValueType *Vx, ValueType *Vy, ValueType *P,
	            	     ValueType *Rx, ValueType *Ry, ValueType *Rc,
	            	     ValueType *Resx, ValueType *Resy, ValueType *Resc,
			     ValueType *Etan, ValueType *Etas,
			     int Row, int Col, int g_Row, int g_Col, 
			     int level);
#endif

#ifdef __GPU__
#define PLATFORM __global__
#else
#define PLATFORM __host__
#endif

__global__ void prolongation( ValueType *M_f, ValueType *M_c, ValueType dx, ValueType dy, bool useEta, ValueType *etanf, ValueType *etanc, ValueType Koef, int level, int LB, int UB, int mode = 0);
__global__ void restriction( ValueType *M_f, ValueType *M_c, ValueType dx, ValueType dy, bool useEta, ValueType *etanf, ValueType *etanc, int level, int LB, int UB);
__global__ void viscosity_restriction( ValueType *etanf, ValueType *etanc, ValueType *etasf, ValueType *etasc, int level);
__global__ void fillGhost( ValueType *g_M, int RB, int LB, int UB, int BB, int level, int mode);
__device__ void fillGhostB( ValueType *s_M, ValueType *g_M, int s_Row, int s_Col, int g_Row, int g_Col, int LB, int RB, int UB, int BB, int level, int debug, int value);

////////////////////////////////
//     General Operations     //
////////////////////////////////
__device__ ValueType get( int Row, int Col, ValueType *M); // Get element from 2D index
__device__ void set( int Row, int Col, ValueType *M, ValueType Val); // Set value to M(Col,Row)
__device__ ValueType getG( int Row, int Col, const ValueType *M, int level); // Get element from 2D index
__device__ void setG( int Row, int Col, ValueType *M, const ValueType Val, int level); // Set value to M(Col,Row)
__device__ ValueType grad( const int Row, const int Col, //! Row and column index
			   const ValueType *M, //! Array to get values from
			   const int xDisplacement, //! Compute gradient along x axis
			   const int yDisplacement, //! Compute gradient along y axis
			   const int level //! Current multigrid level
			   );
__device__ ValueType gradVelocity( const int Row, const int Col, //! Row and column index
				   const ValueType *M, //! Array to get values from
				   const int xDisplacement, //! Compute gradient along x axis
				   const int yDisplacement, //! Compute gradient along y axis
				   const int level //! Current multigrid level
				   );
__device__ void upG( int Row, int Col, ValueType *M, const ValueType Val, int level); // Set value to M(Col,Row)
__device__ void getGhost( ValueType *s_M, ValueType *g_M, int s_Row, int s_Col, int g_id, int level); // Get ghost nodes
__device__ void setGhost( ValueType *s_M, ValueType *g_M, int Row, int Col, int g_Row, int g_Col, int LB, int RB, int UB, int BB, int level); // Set shared ghost nodes to global

__global__ void setBoundaries(ValueType *g_Vx, ValueType *g_Vy, ValueType *g_P, int level);

#endif

// CUDA SDK Templates

// Utility class used to avoid linker errors with extern
// unsized shared memory arrays with templated type
template<class T>
struct SharedMemory
{
    __device__ inline operator       T*()
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }

    __device__ inline operator const T*() const
    {
        extern __shared__ int __smem[];
        return (T*)__smem;
    }
};

// specialize for double to avoid unaligned memory 
// access compile errors
template<>
struct SharedMemory<double>
{
    __device__ inline operator       double*()
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }

    __device__ inline operator const double*() const
    {
        extern __shared__ double __smem_d[];
        return (double*)__smem_d;
    }
};
