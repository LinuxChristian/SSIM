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

//    Multigrid solver using Gauss-Seidel smoothing using the layout from Gerya, 2010
//
//    Contact:
//      Christian Brædstrup
//      christian.fredborg@geo.au.dk
//      Ice, Water and Sediment Group
//      Institut for Geoscience
//      University of Aarhus 

#ifdef __GPU__
   #include "config.cuh"
   #include "cuPrintf.cu"

  __constant__ ConstantMem constMem;
#else

#define maxLevel 8    // Max number of levels
#define BLOCK_SIZE 32 // Size of shared memory to allocate
#define PADDING 1     // Thickness of ghost nodes

typedef double ValueType;

#include "constMemStruct.h"
#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>
#include <cmath>
//#include <omp.h> // OpenMP

ConstantMem constMem;
#endif


#ifdef __GPU__
// Get element of the array M that has M(Col,Row)
__device__ ValueType get( int Row, int Col, ValueType *M) {
  return M[Row*(blockDim.x+2)+Col];
};

// Set element to M(Col,Row)
__device__ void set( int Row, int Col, ValueType *M, ValueType Val) {
  M[Row*(blockDim.x+2)+Col] = Val;
};
#endif

#ifdef __GPU__               
__device__ 
#endif
inline ValueType getG( int Row, int Col, const ValueType *M, int level)
{
  // Get element of the array M that has M(Col,Row)
  return M[Row*(constMem.XNum[level])+Col];
};

#ifdef __GPU__               
__device__
#endif
inline void setG( int Row, int Col, ValueType *M, const ValueType Val, int level)
{
  // Set element to M(Col,Row)
  M[Row*(constMem.XNum[level])+Col] = Val;
};

#ifdef __GPU__               
__device__
#endif
void upG( int Row, int Col, ValueType *M, const ValueType Val, int level)
{
  // Update element to M(Col,Row)
  M[Row*(constMem.XNum[level])+Col] += Val;
};


/**
 * Computes finite difference gradient for all center variables
 *
 *
 * \verbatim embed:rst
 * .. seealso:: gradVelocity
 * \endverbatim
 *
 * @param Row Global Row
 * @param Col Global Column
 * @param M Array to get values from
 * @param xDisplacement The displacement along the x-axis (in whole units i.e. 0 or 1)
 * @param yDisplacement The displacement along the y-axis (in whole units i.e. 0 or 1)
 * @param level Current multigrid level
*/
#ifdef __GPU__               
__device__
#endif
ValueType grad( const int Row, const int Col, //! Row and column index
		const ValueType *M, //! Array to get values from
		const int xDisplacement, //! Compute gradient along x axis
		const int yDisplacement, //! Compute gradient along y axis
		const int level //! Current multigrid level
		) {

  ValueType interp = 0.0;  

  ValueType interp1 = 0.0;
  ValueType interp2 = 0.0;

  // Gradient 1
  interp1 = ( (getG(Row+yDisplacement, Col+xDisplacement, M, level)) - (getG(Row, Col, M, level)) )/(((ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]));

  // Gradient 2
  interp2 = ( (getG(Row, Col, M, level)) - (getG(Row-yDisplacement, Col-xDisplacement, M, level)) )/(((ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]));

  interp=(interp1+interp2)*0.5;

    
  /*
  interp = ( 
	      (getG(Row+yDisplacement, Col+xDisplacement, M, level)) 
	    - (getG(Row-yDisplacement, Col-xDisplacement, M, level)) )
    /(2.0 * (
	       (ValueType)(xDisplacement)*constMem.Dx[level] 
	     + (ValueType)(yDisplacement)*constMem.Dy[level])
      );
 */
  
  return interp;
};

/**
 * Compute gradient of velocities using finite difference
 *
 * @param Row Global Row
 * @param Col Global Column
 * @param M Array to get values from
 * @param xDisplacement The displacement along the x-axis
 * @param yDisplacement The displacement along the y-axis
 * @param level Current multigrid level
*/
#ifdef __GPU__               
__device__
#endif
ValueType gradVelocity( const int Row, const int Col, //! Row and column index
			const ValueType *M, //! Array to get values from
			const int xDisplacement, //! Compute gradient along x axis
			const int yDisplacement, //! Compute gradient along y axis
			const int level //! Current multigrid level
			) {

  ValueType interp = 0.0;
  //  ValueType interp2 = 0.0;
  interp = ( (getG(Row+yDisplacement, Col+xDisplacement, M, level)) - (getG(Row, Col, M, level)) )/( (ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]);

  return interp;
};

#ifdef __GPU__
// GPU Device specific funtions

////////////////////////////////////////////////////////////////////////////////
//                 setGhost				                      //
//                               			                      //
//  DESCRIPTION:                           		                      //
//  Copies ghost nodes from shared memory to global                           //
//  memory to share with other blocks  			                      //
//                               			                      //
//  INPUT:                                                                    //
//  s_M   - Shared matrix                                                     // 
//  Row   - Shared row                                                        //
//  Col   - Shared column                                                     //
//  g_Row - Global row                                                        //
//  g_Col - Global column                                                     //
//  level - Current level                                                     //
//  RB    - Right boundary                                                    //
//  LB    - Left boundary                                                     //
//  UB    - Upper boundary                                                    //
//  BB    - Bottom boundary                                                   //
//                               			                      //
//  OUTPUT:                                                                   //
//  g_M   - Global matrix                                                     //
////////////////////////////////////////////////////////////////////////////////
__device__ void setGhost( ValueType *s_M, ValueType *g_M, int Row, int Col, int g_Row, int g_Col, int LB, int RB, int UB, int BB, int level) {

  // Copy global to shared
  if (Row==UB) {
    // Halos above
    set(Row-1,Col,s_M,getG(g_Row-1,g_Col,g_M,level));
  };

  if (Row==BB) { 
    // Halos below
    set(Row+1,Col,s_M,getG(g_Row+1,g_Col,g_M,level));
  };

  if (Col==LB) { 
    // Halos left
    set(Row,Col-1,s_M,getG(g_Row,g_Col-1,g_M,level));
  };

  if (Col==RB) { 
    // Halos right
    set(Row,Col+1,s_M,getG(g_Row,g_Col+1,g_M,level));
  };
  __syncthreads();
  
};


////////////////////////////////////////////////////////////
//                 fillGhostB  				  //
//                               			  //
//  DESCRIPTION:                           		  //
//  Fills the outer most ghost nodes with the value       //
//  from the outer level compute nodes on a block         //
//  level                                                 //
//                               			  //
//  NOTE:                                                 //  
//  If boundary is set to 0 it will not be set            //
//                               			  //
//  INPUT:                                                //
//  S_M   - Pointer to matrix in shared                   //
//  g_M   - Pointer to matrix in global                   //
//  s_Row - Shared Row                                    //
//  s_Col - Shared Column                                 //
//  g_Row - Global Row                                    //
//  g_Col - Global Column                                 //
//  RB    - Right boundary                                //
//  LB    - Left boundary                                 //
//  UB    - Upper boundary                                //
//  BB    - Bottom boundary                               //
//  level - Current level                                 //
//  debug - Bit to enable debug                           //
//  value - Value to set in debugging                     //
//                               			  //
//  OUTPUT:                                               //
//  None                                                  //
////////////////////////////////////////////////////////////
__device__ void fillGhostB( ValueType *s_M, ValueType *g_M, int s_Row, int s_Col, int g_Row, int g_Col, int LB, int RB, int UB, int BB, int level, int debug, int value) {

  // LOAD WALLS
  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    /*
    /////////////////////////////////////////
    // update the data slice in smem
    if(threadIdx.y<PADDING)
      // halo above/below
      {
	set(threadIdx.y                   , s_Col, s_M, getG(g_Row-PADDING   , g_Col, g_M, level));
	set(threadIdx.y+blockDim.y+PADDING, s_Col, s_M, getG(g_Row+blockDim.y, g_Col, g_M, level));
      }
    if(threadIdx.x<PADDING)
      // halo left/right
      {
	set(s_Row, threadIdx.x                   , s_M, getG(g_Row, g_Col-PADDING   , g_M, level));
	set(s_Row, threadIdx.x+blockDim.x+PADDING, s_M, getG(g_Row, g_Col+blockDim.y, g_M, level));
      }
    
    */
    
  // Upper boundary
  if (s_Row==UB && UB!=0) { 
  // Halos above
    set(s_Row-1,s_Col,s_M, (debug)?(value):(getG(g_Row-1,g_Col,g_M,level)) );
  };

  // Bottom boundary
  if ( ( g_Row == constMem.YNum[level]-2 || s_Row==BB ) && BB!=0) { 
    // Halos below
    set(s_Row+1,s_Col,s_M, (debug)?(value):(getG(g_Row+1,g_Col,g_M,level)) );
  };

  // Left boundary
  if (s_Col==LB && LB!=0) { 
    // Halos left
    set(s_Row,s_Col-1,s_M, (debug)?(value):(getG(g_Row,g_Col-1,g_M,level)) );
  };

  // Right boundary
  if ( ( s_Col==RB || g_Col == constMem.XNum[level]-2 ) && RB!=0) { 
    // Halos right
    set(s_Row,s_Col+1,s_M, (debug)?(value):(getG(g_Row,g_Col+1,g_M,level)) );
  };
   
    
  // LOAD CORNERS
  // Load upper left corners
  // Done by upper left thread
  if (s_Col == LB && s_Row == UB ) {
    set(s_Row-1,s_Col-1,s_M, (debug)?(value):(getG(g_Row-1,g_Col-1,g_M,level)) );
  };

  // Load upper right corners
  // Done by upper right thread
  if ( ( s_Col == RB || g_Col == constMem.XNum[level]-2 ) && s_Row == UB  ) {
    set(s_Row-1,s_Col+1,s_M, (debug)?(value):(getG(g_Row-1,g_Col+1,g_M,level)) );
  };

  // Load lower left corners
  // Done by lower left thread
  if ( s_Col == LB && ( s_Row == BB || g_Row == constMem.YNum[level]-2 ) ) {
    set(s_Row+1,s_Col-1,s_M, (debug)?(value):(getG(g_Row+1,g_Col-1,g_M,level)) );
  };

  // Load lower right corners
  // Done by lower right thread
  if ( (s_Col == RB || g_Col == constMem.XNum[level]-2) && ( s_Row == BB || g_Row == constMem.YNum[level]-2 ) ) {
    set(s_Row+1,s_Col+1,s_M, (debug)?(value):(getG(g_Row+1,g_Col+1,g_M,level)) );
  };

  };
};
#endif


////////////////////////////////////////////////////////////
//                 fillGhost  				  //
//                               			  //
//   DESCRIPTION:                           		  //
//   Fills the outer most ghost nodes with the value      //
//   from the outer level compute nodes 		  //
//                               			  //
//  INPUT:                                                //
//  g_M   - Pointer to matrix in global                   //
//  RB    - Right boundary                                //
//  LB    - Left boundary                                 //
//  UB    - Upper boundary                                //
//  BB    - Bottom boundary                               //
//  level - Current level                                 //
//                               			  //
//  OUTPUT:                                               //
//  None                                                  //
////////////////////////////////////////////////////////////
#ifdef __GPU__
__global__ 
#endif
void fillGhost( ValueType *g_M, int LB, int RB, int UB, int BB, int level, int mode) {

	  
  //  int g_Col =  blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  //  int g_Row =  blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  //  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

  switch (mode) {

  case 1:
  // Restrict g_Row and g_Col to currect dimensions 
  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    if ( g_Row==UB && UB!=0) { 
      // Halos above
      setG(g_Row-1,g_Col,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    if ( g_Row==BB && BB!=0) { 
      // Halos below
      setG(g_Row+1,g_Col,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    if (g_Col==LB && LB!=0) { 
      // Halos left
      setG(g_Row,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    if (g_Col==RB && RB!=0) { 
      // Halos right
      setG(g_Row,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // LOAD CORNERS
    
    // Load upper left corners
    // Done by upper left thread
    if (g_Col == LB && g_Row == UB ) {
      setG(g_Row-1,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load upper right corners
    // Done by upper right thread
    if (g_Col == RB && g_Row == UB ) {
      setG(g_Row-1,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load lower left corners
    // Done by lower left thread
    if (g_Col == LB && g_Row == BB ) {
      setG(g_Row+1,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load lower right corners
    // Done by lower right thread
    if (g_Col == RB && g_Row == BB ) {
      setG(g_Row+1,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
  };

  break;
  case 0:
    // Fill ghost nodes like boundries
    // so 0 is filled with XNum-1 and similar
  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    // UB, BB, LB, RB must be given as the row efter or before the
    // one that is to be filled
    if ( g_Row==UB && UB!=0) { 
      // Halos above
      setG( g_Row-1, g_Col, g_M, getG( BB, g_Col, g_M, level), level);
      //      setG(g_Row-1,g_Col,g_M,getG(,g_Col,g_M,level),level);
    };
    
    if ( g_Row==BB && BB!=0) { 
      // Halos below
      setG( g_Row+1, g_Col, g_M, getG( UB, g_Col, g_M, level), level);
      //      setG(g_Row+1,g_Col,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    if (g_Col==LB && LB!=0) { 
      // Halos left
      setG( g_Row, g_Col-1, g_M, getG( RB, g_Col, g_M, level), level);
      //      setG(g_Row,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    if (g_Col==RB && RB!=0) { 
      // Halos right
      setG( g_Row, g_Col+1, g_M, getG( LB, g_Col, g_M, level), level);
      //      setG(g_Row,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // LOAD CORNERS
    /*
    // Load upper left corners
    // Done by upper left thread
    if (g_Col == LB && g_Row == UB ) {
      setG(g_Row-1,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load upper right corners
    // Done by upper right thread
    if (g_Col == RB && g_Row == UB ) {
      setG(g_Row-1,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load lower left corners
    // Done by lower left thread
    if (g_Col == LB && g_Row == BB ) {
      setG(g_Row+1,g_Col-1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    
    // Load lower right corners
    // Done by lower right thread
    if (g_Col == RB && g_Row == BB ) {
      setG(g_Row+1,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };
    */
  };
    break;
  case 2:

    if ( g_Row== constMem.YNum[level] - 2) { 
      // Halos below
      setG(g_Row+1,g_Col,g_M,getG(g_Row,g_Col,g_M,level),level);
    };

    if (g_Col== constMem.XNum[level] - 2) { 
      // Halos right
      setG(g_Row,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };

    if (g_Col== constMem.XNum[level] - 2 && g_Row == constMem.XNum[level] - 2) { 
      // Halos right
      setG(g_Row+1,g_Col+1,g_M,getG(g_Row,g_Col,g_M,level),level);
    };

    break;
    
  };

#ifndef __GPU__
      }}
#endif
};


/*
 * General Operations
 */
#ifdef __GPU__
__global__ 
#endif
void Restrict(const ValueType* FineGrid, ValueType* CoarseGrid, const int CoarseLevel, int UB, int LB, const int mode) {
  // NOTE: Only works when dim_c = dim_f/2 + 1!

  #ifdef __GPU__
  int Col_c = blockIdx.x*blockDim.x+threadIdx.x+PADDING+LB;
  int Row_c = blockIdx.y*blockDim.y+threadIdx.y+PADDING+UB; 
  #else
    for (int Col_c=PADDING+LB;Col_c<constMem.XNum[CoarseLevel]-2;Col_c++) {
      for (int Row_c=PADDING+UB;Row_c<constMem.YNum[CoarseLevel]-2;Row_c++) {
  #endif

  int Col_f = 2*(Col_c-LB)-1+LB;
  int Row_f = 2*(Row_c-UB)-1+UB;
  ValueType Interp = 0.0;

  // Interior nodes
  if ( Col_c < constMem.XNum[CoarseLevel]-2 && Row_c < constMem.YNum[CoarseLevel]-2) {

    if (mode) {
      // Half weighted
      Interp = 0.5 * getG(Row_f, Col_f, FineGrid, CoarseLevel-1)  
	+  0.125 * (   getG(Row_f + 1, Col_f    , FineGrid, CoarseLevel-1)
		       + getG(Row_f    , Col_f + 1, FineGrid, CoarseLevel-1)
		       + getG(Row_f - 1, Col_f    , FineGrid, CoarseLevel-1)
		       + getG(Row_f    , Col_f - 1, FineGrid, CoarseLevel-1) );
    } else {
      // Full weighted
      Interp = 0.25 * getG(Row_f, Col_f, FineGrid, CoarseLevel-1)  
	+ 0.125 * (   getG(Row_f + 1, Col_f    , FineGrid, CoarseLevel-1)
		      + getG(Row_f    , Col_f + 1, FineGrid, CoarseLevel-1)
		      + getG(Row_f - 1, Col_f    , FineGrid, CoarseLevel-1)
		      + getG(Row_f    , Col_f - 1, FineGrid, CoarseLevel-1) )
	+ 0.0625 * (  getG(Row_f + 1, Col_f +1   , FineGrid, CoarseLevel-1)
		      + getG(Row_f + 1, Col_f - 1   , FineGrid, CoarseLevel-1)
		      + getG(Row_f - 1, Col_f + 1   , FineGrid, CoarseLevel-1)
		      + getG(Row_f - 1, Col_f -1   , FineGrid, CoarseLevel-1));
    };
							     
    setG(Row_c, Col_c, CoarseGrid, Interp, CoarseLevel);
  };

  /* Not very effective */
  
  // Boundary nodes - Straight injection
  if (Col_c == 1+LB && Row_c < constMem.YNum[CoarseLevel]-2) {
    // Left nodes
    Interp = 0.125*getG(Row_f-1, 1+LB+1, FineGrid, CoarseLevel-1)
           + 0.125*getG(Row_f-1, 1+LB  , FineGrid, CoarseLevel-1)
           + 0.250*getG(Row_f  , 1+LB  , FineGrid, CoarseLevel-1)
           + 0.250*getG(Row_f  , 1+LB+1, FineGrid, CoarseLevel-1)
           + 0.125*getG(Row_f+1, 1+LB  , FineGrid, CoarseLevel-1)
           + 0.125*getG(Row_f+1, 1+LB+1, FineGrid, CoarseLevel-1);

    setG(Row_c, 1+LB, CoarseGrid, Interp, CoarseLevel);
    
    Interp = 0.125*getG(Row_f-1, constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1)
           + 0.125*getG(Row_f-1, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1)
           + 0.250*getG(Row_f, constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1) 
           + 0.250*getG(Row_f, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1) 
           + 0.125*getG(Row_f+1, constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1) 
           + 0.125*getG(Row_f+1, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1);

    // Right nodes
    setG(Row_c, constMem.XNum[CoarseLevel]-2, CoarseGrid, Interp, CoarseLevel);
  };
  
  if (Row_c == 2+UB && Col_c < constMem.YNum[CoarseLevel]-2) {
    // Upper nodes
    Interp = 0.125*getG(1+UB, Col_f-1, FineGrid, CoarseLevel-1)
           + 0.125*getG(2+UB, Col_f-1, FineGrid, CoarseLevel-1)
           + 0.250*getG(1+UB, Col_f  , FineGrid, CoarseLevel-1)
           + 0.250*getG(2+UB, Col_f  , FineGrid, CoarseLevel-1)
           + 0.125*getG(1+UB, Col_f+1, FineGrid, CoarseLevel-1)
           + 0.125*getG(2+UB, Col_f+1, FineGrid, CoarseLevel-1);

    setG(1+UB, Col_c, CoarseGrid, Interp, CoarseLevel);

    Interp = 0.125*getG(constMem.YNum[CoarseLevel-1]-2, Col_f-1, FineGrid, CoarseLevel-1)
           + 0.125*getG(constMem.YNum[CoarseLevel-1]-3, Col_f-1, FineGrid, CoarseLevel-1)
           + 0.250*getG(constMem.YNum[CoarseLevel-1]-2, Col_f  , FineGrid, CoarseLevel-1)
           + 0.250*getG(constMem.YNum[CoarseLevel-1]-3, Col_f  , FineGrid, CoarseLevel-1)
           + 0.125*getG(constMem.YNum[CoarseLevel-1]-2, Col_f+1, FineGrid, CoarseLevel-1)
           + 0.125*getG(constMem.YNum[CoarseLevel-1]-3, Col_f+1, FineGrid, CoarseLevel-1);
    // Lower
    setG(constMem.YNum[CoarseLevel]-2, Col_c, CoarseGrid, Interp, CoarseLevel);

    // Corner Nodes
    if (Col_c == 2+LB) {
      /* Upper Left */
      Interp = 0.25 * getG(1+UB, 1+LB, FineGrid, CoarseLevel-1)
   	     + 0.25 * getG(2+UB, 1+LB, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(1+UB, 2+LB, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(2+UB, 2+LB, FineGrid, CoarseLevel-1);

      setG(1+UB, 1+LB, CoarseGrid, Interp, CoarseLevel);

      /* Lower Left*/
      Interp = 0.25 * getG(constMem.YNum[CoarseLevel-1]-2, 1+LB, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-2, 2+LB, FineGrid, CoarseLevel-1)
  	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-3, 1+LB, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-3, 2+LB, FineGrid, CoarseLevel-1);
      
      setG(constMem.YNum[CoarseLevel]-2, 1+LB, CoarseGrid, Interp, CoarseLevel);

      /* Upper Right */
      Interp = 0.25 * getG(1+UB  , constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1)
   	     + 0.25 * getG(1+UB+1, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(1+UB  , constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(1+UB+1, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1);

      setG(1+UB, constMem.XNum[CoarseLevel]-2, CoarseGrid, Interp, CoarseLevel);

      /* Lower Right */
      Interp = 0.25 * getG(constMem.YNum[CoarseLevel-1]-2, constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-2, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-3, constMem.XNum[CoarseLevel-1]-2, FineGrid, CoarseLevel-1)
	     + 0.25 * getG(constMem.YNum[CoarseLevel-1]-3, constMem.XNum[CoarseLevel-1]-3, FineGrid, CoarseLevel-1);

      setG(constMem.YNum[CoarseLevel]-2, constMem.XNum[CoarseLevel]-2, CoarseGrid, Interp, CoarseLevel);
    }
  };
  
 #ifndef __GPU__
      }
    }
 #endif
};


#ifdef __GPU__
__global__ 
#endif
void Prolong( ValueType* FineGrid, const ValueType* CoarseGrid, ValueType Koef, int Nodes, int FineLevel, int UB, int LB ) {
  // Only works when dim_c = dim_f/2 +1

  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 
  #else
  for (int g_Col=0;g_Col<constMem.XNum[FineLevel]-1;g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[FineLevel]-1;g_Row++) {
  #endif  

  int f_Row, f_Col, c_Col, c_Row;
  ValueType interp = 0.0;
  ValueType Initial = 0.0;
  ValueType WeightOldValue = 0.0;

  if (Nodes == 0) {
    // Do all overlapping nodes
    f_Row = g_Row*2 + 1 + UB;
    f_Col = g_Col*2 + 1 + LB; // Overlap nodes
    #ifdef __GPU__
      c_Col = __double2int_ru(f_Col/2);
      c_Row = __double2int_ru(f_Row/2);
    #else
      c_Col = (int)(ceil(f_Col/2));
      c_Row = (int)(ceil(f_Row/2));
    #endif

    if (c_Col < 1)
      c_Col = 1;

    if (c_Row < 1)
      c_Row = 1;

    if ( f_Row < constMem.YNum[FineLevel]-1  && f_Col < constMem.XNum[FineLevel]-1) {
      Initial = getG(f_Row, f_Col, FineGrid, FineLevel);
      interp = getG(c_Row, c_Col, CoarseGrid, FineLevel+1);
      
      setG(f_Row, f_Col, FineGrid, WeightOldValue*Initial + interp*Koef, FineLevel);
    };

  } else if (Nodes == 1) {
    // Do vertical and horisontal interpolation
    int  IsEvenRow, IsEvenCol;
      
    f_Row = g_Row + 1 + UB;
    f_Col = g_Col + 1 + LB;

    IsEvenRow = (f_Row+UB) % 2;
    IsEvenCol = (f_Col+LB) % 2;


    if (f_Row < constMem.YNum[FineLevel]-1 && f_Col < constMem.XNum[FineLevel]-1 && IsEvenRow != IsEvenCol) {

      #ifdef __GPU__
        c_Col = __double2int_rd(f_Col / 2);
        c_Row = __double2int_rd(f_Row / 2);
      #else
        c_Col = (int)(floor(f_Col / 2));
        c_Row = (int)(floor(f_Row / 2));
      #endif

    if (IsEvenRow == 0 && IsEvenCol == 1) {
      // Vertical Interpolation
      c_Col += 1;
      interp = 0.5 * ( getG( c_Row, c_Col, CoarseGrid, FineLevel+1 )
		     + getG( c_Row+1, c_Col, CoarseGrid, FineLevel+1 ) );
      
      Initial = getG(f_Row, f_Col, FineGrid, FineLevel);
      setG(f_Row, f_Col, FineGrid, WeightOldValue*Initial+interp*Koef, FineLevel);
    };

    if (IsEvenRow == 1 && IsEvenCol == 0) {
      // Vertical Interpolation
      c_Row += 1;
      interp = 0.5 * ( getG( c_Row, c_Col, CoarseGrid, FineLevel+1 )
		     + getG( c_Row, c_Col+1, CoarseGrid, FineLevel+1 ) );

      Initial = getG(f_Row, f_Col, FineGrid, FineLevel);
      setG(f_Row, f_Col, FineGrid, WeightOldValue*Initial+interp*Koef, FineLevel);

    };
    };

  } else if (Nodes == 2) {
    // Interpolate center nodes from other fine nodes

    f_Row = 2*g_Row + 2 + UB;
    f_Col = 2*g_Col + 2 + LB; // Overlap nodes

    #ifdef __GPU__
    c_Row = __double2int_rd( (f_Row + 1)/2 ) + 1;
    c_Col = __double2int_rd( (f_Col + 1)/2 ) + 1;
    #else
    c_Row = (int)(floor( (f_Row + 1)/2 ) + 1);
    c_Col = (int)(floor( (f_Col + 1)/2 ) + 1);
    #endif

    if ( f_Row < constMem.YNum[FineLevel]-1 && f_Col < constMem.XNum[FineLevel]-1 ) {
      Initial = getG(f_Row, f_Col, FineGrid, FineLevel);
      interp = 0.25 * (    getG(c_Row    , c_Col    , CoarseGrid, FineLevel+1)
			   + getG(c_Row - 1, c_Col    , CoarseGrid, FineLevel+1)
			 + getG(c_Row    , c_Col - 1, CoarseGrid, FineLevel+1)
			 + getG(c_Row - 1, c_Col - 1, CoarseGrid, FineLevel+1) );

      setG(f_Row, f_Col, FineGrid, WeightOldValue*Initial + interp*Koef, FineLevel);
    };    
  };

 #ifndef __GPU__
      }
    }
 #endif

  return;

};


#ifdef __GPU__
__global__ 
#endif
void ResetLevel(ValueType *g_M, ValueType Value, int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 
  if (g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level]) {
  #else
    for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
  #endif
  
	if ( g_Col < constMem.XNum[level] && g_Row < constMem.YNum[level]) {
	  setG(g_Row, g_Col, g_M, 0.0, level);
	};
      };

  #ifndef __GPU__
    };
  #endif
};

/**
 * Reset the current vector or matrix with the value given
 *
 * @param g_M Pointer to global data to reset
 * @param Value The value to give g_M
 * @param XDim Dimension of memory in x
 * @param YDim Dimension of memory in y
 * @param level Current multigrid level
*/
#ifdef __GPU__
__global__ 
#endif
void ResetLevel(ValueType *g_M, ValueType Value, const unsigned int XDim, const unsigned int YDim, int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 

  if ( g_Col < XDim && g_Row < YDim) {
  #else
    for (unsigned int g_Col=0;g_Col<XDim;g_Col++) {
      for (unsigned int g_Row=0;g_Row<YDim;g_Row++) {
  #endif
  
	setG(g_Row, g_Col, g_M, Value, level);
      };

  #ifndef __GPU__
    };
  #endif
};

////////////////////////////////////////////////////////////////////////////////
//                       prolongation		                              //
//                               			                      //
//  DESCRIPTION:                           		                      //
//  Interpolates values from the coarse grid to finer grids                   //
//                               			                      //
//  INPUT:                                                                    //
//  M_c    - Coarse grid                                                      //
//  dx     - Displacement of grid in x direction                              //
//  dy     - Displacement of grid in y direction                              //
//  useEta - Update using local eta                                           //
//  etanf  - Viscosity on fine grid                                           //
//  etanc  - Viscosity on coarse grid                                         //
//  Koef   - Coefficient of update term                                       //
//  level  - Current level (Coarse level)                                     //
//                               			                      //
//  OUTPUT:                                                                   //
//  M_f    - Fine gird                                                        //
//                               			                      //
//  NOTE:                             			                      //
//  Grids are staggered but when interpolating only the relative distance     //
//  is used and therefore grids can be considdered non-staggered.             //
//                               			                      //
////////////////////////////////////////////////////////////////////////////////
#ifdef __GPU__
__global__ 
#endif
void prolongation( ValueType *M_f, ValueType *M_c, ValueType dx, ValueType dy, bool useEta, ValueType *etanf, ValueType *etanc, ValueType Koef, int CoarseLevel, int LB, int UB, int mode)
{
  // Level is fine
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[CoarseLevel-1]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[CoarseLevel-1]-1;g_Row++) {
  #endif

  const int Row_f = g_Row+UB;
  const int Col_f = g_Col+LB;
  int OldWeight = 1.0; // Note: Change to 1.0

  if ( Row_f < constMem.YNum[CoarseLevel-1]-1 && Col_f < constMem.XNum[CoarseLevel-1]-1 ) {
    
    int Row_c, Col_c; // Position in both arrays
    ValueType xf, yf, x1, y1, dxdx, dydy; // Physical position

    xf=(Col_f + dx - PADDING)*constMem.Dx[CoarseLevel-1];
    yf=(Row_f + dy - PADDING)*constMem.Dy[CoarseLevel-1];

    #ifdef __GPU__
      Col_c = __double2int_rn(xf/constMem.Dx[CoarseLevel]);
      Row_c = __double2int_rn(yf/constMem.Dy[CoarseLevel]);
    #else
      Col_c = (int) (round(xf/constMem.Dx[CoarseLevel]));
      Row_c = (int) (round(yf/constMem.Dy[CoarseLevel]));
    #endif

    x1 = (Col_c + dx - PADDING)*constMem.Dx[CoarseLevel];
    y1 = (Row_c + dy - PADDING)*constMem.Dy[CoarseLevel];
    
    if (Col_c < 1+LB)
      Col_c = 1+LB;

    if (Row_c < 1+UB)
      Row_c = 1+UB;

    if (Col_c > constMem.XNum[CoarseLevel]-2)
      Col_c = constMem.XNum[CoarseLevel]-2;

    if (Row_c > constMem.YNum[CoarseLevel]-2)
      Row_c = constMem.YNum[CoarseLevel]-2;


    ValueType dR = 0.0;
    
    dxdx = (xf-x1)/constMem.Dx[CoarseLevel];
    dydy = (yf-y1)/constMem.Dy[CoarseLevel];

    if (Row_c < constMem.YNum[CoarseLevel]-1 && Col_c < constMem.XNum[CoarseLevel]-1) {
      
      dR = (1 - dxdx) * (1 - dydy) * getG(Row_c  ,Col_c  ,M_c,CoarseLevel) // Upper Left
	 + (1 - dxdx) * (    dydy) * getG(Row_c+1,Col_c  ,M_c,CoarseLevel) // Lower Left
	 + (    dxdx) * (1 - dydy) * getG(Row_c  ,Col_c+1,M_c,CoarseLevel) // Upper Right
	 + (    dxdx) * (    dydy) * getG(Row_c+1,Col_c+1,M_c,CoarseLevel); // Lower Right
    }
    
    ValueType Old = getG(Row_f,Col_f,M_f,CoarseLevel-1);

    setG(Row_f, Col_f, M_f, (ValueType) OldWeight*Old+(Koef*dR), CoarseLevel-1);
    };

  #ifndef __GPU__
      }
    }
  #endif

};




////////////////////////////////////////////////////////////
//            Restriction  				  //
//                               			  //
//  DESCRIPTION:                           		  //
//  General restriction rutine. First index in both       //
//  fine and coarse grid is (1,1) but when computing      //
//  physical position indexes start at (0,0)     	  //
//                               			  //
//  INPUT:                                                //
//  M_f     - Fine grid                                   //
//  dx      - X-displacement of grid point                //
//  dy      - Y-displacement of grid point                //
//  useEtan - Correct pressure with local etan            //
//  etanf   - Normal viscosity fine grid                  //
//  etanc   - Normal viscosity coarse grid                //
//  level   - Current level (coarse grid)                 //
//                               			  //
//  OUTPUT:                                               //
//  M_c     - Coarse grid                                 //
//                               			  //
//  NOTE:                             			  //
//  Grids are staggered but when interpolating   	  //
//  only the relative distance is used and therefore	  //
//  grids can be considdered non-staggered.      	  //
////////////////////////////////////////////////////////////
#ifdef __GPU__
__global__ 
#endif
void restriction(ValueType *M_f, ValueType *M_c, 
		 ValueType dx, ValueType dy, 
		 bool useEta, ValueType *etanf, ValueType *etanc, 
		 int CoarseLevel, int LB, int UB)
{
  // Block row and col
  #ifdef __GPU__
  int s_Col = threadIdx.x+PADDING+LB; 
  int s_Row = threadIdx.y+PADDING+UB; 
  // Constant index to current coarse node
  const int c_Col = blockIdx.x*blockDim.x+s_Col; 
  const int c_Row = blockIdx.y*blockDim.y+s_Row;

  #else
    for (int c_Col=PADDING+LB;c_Col<constMem.XNum[CoarseLevel]-1;c_Col++) {
      for (int c_Row=PADDING+UB;c_Row<constMem.YNum[CoarseLevel]-1;c_Row++) {
  #endif

  const int xRatio = constMem.Dx[CoarseLevel-1]/constMem.Dx[CoarseLevel]-1;
  const int yRatio = constMem.Dy[CoarseLevel-1]/constMem.Dy[CoarseLevel]-1;

  if ( c_Row < constMem.YNum[CoarseLevel]-1 && c_Col < constMem.XNum[CoarseLevel]-1 ) {

    // Taras style marker interpolation
    int f_Row, f_Col; 
    ValueType xc, yc, xf, yf;
    ValueType WSum = 0.0, W = 0.0;
    ValueType dR = 0.0;

    // Physical location of coarse basenode
    xc = (c_Col + dx - LB)*constMem.Dx[CoarseLevel];
    yc = (c_Row + dy - UB)*constMem.Dy[CoarseLevel];

    // loop over all fine nodes that are within one
    // coarse grid spaceing
    for (int i = -xRatio; i <= xRatio+1; i++) {
          for (int j = -yRatio; j <= xRatio+1; j++) {
	    
	    // Find closest fine node with lowest index
	    #ifdef __GPU__
	      f_Col = __double2int_rd((xc + i*constMem.Dx[CoarseLevel-1])/constMem.Dx[CoarseLevel-1])+LB;
	      f_Row = __double2int_rd((yc + j*constMem.Dy[CoarseLevel-1])/constMem.Dy[CoarseLevel-1])+UB;
	    #else
	      f_Col = (int)(floor((xc + i*constMem.Dx[CoarseLevel-1])/constMem.Dx[CoarseLevel-1])+LB);
 	      f_Row = (int)(floor((yc + j*constMem.Dy[CoarseLevel-1])/constMem.Dy[CoarseLevel-1])+UB);
	    #endif

	    if (f_Col > constMem.XNum[CoarseLevel-1]-2)
	      f_Col = constMem.XNum[CoarseLevel-1]-2;
	      
	    if ( f_Row > constMem.YNum[CoarseLevel-1]-2)
	      f_Row = constMem.YNum[CoarseLevel-1]-2;
	    
	    // Physical location of fine nodes
	    xf = (f_Col + dx - LB)*constMem.Dx[CoarseLevel-1];
	    yf = (f_Row + dy - UB)*constMem.Dy[CoarseLevel-1];

	    W = (1 - abs(xf-xc)/constMem.Dx[CoarseLevel]) * (1 - abs(yf-yc)/constMem.Dy[CoarseLevel]);
	    WSum += W;
	    
	    if (useEta) {
	      dR +=  W * getG(f_Row, f_Col, M_f, CoarseLevel-1) * getG(f_Row, f_Col, etanf, CoarseLevel-1);
	    } else {
	      dR +=  W * getG(f_Row, f_Col, M_f, CoarseLevel-1);
	    };
	  };
    };

    if (useEta) {
      setG(c_Row, c_Col, M_c, dR/WSum/getG(c_Row, c_Col, etanc, CoarseLevel), CoarseLevel);
    } else {
      setG(c_Row, c_Col, M_c, dR/WSum, CoarseLevel);
    };

  }

  #ifndef __GPU__
      };
    };
  #endif
};


/**
 * Preformes reduction of input matrix.
 *
 * Modified implementation of reduce3 algorithm from CUDA SDK.
 * This is not the most optimized algorithm 
 * but it would be overkill to use a more optimized.
 *
 * @param g_idata Global pointer to input
 * @param g_odata Global pointer to output
 * @param n Number of elements to reduce
*/
#ifdef __GPU__
__global__ void reduction(const ValueType *g_idata, ValueType *g_odata, const int level)
{
    ValueType *sdata = SharedMemory<ValueType>();

    // perform first level of reduction,
    // reading from global memory, writing to shared memory
    unsigned int n = constMem.XNum[level]*constMem.YNum[level];
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x*(blockDim.x*2) + threadIdx.x;

    ValueType mySum = (i < n) ? g_idata[i] : 0;
    if (i + blockDim.x < n) 
        mySum += g_idata[i+blockDim.x];  

    sdata[tid] = mySum;
    __syncthreads();

    // do reduction in shared mem
    for(unsigned int s=blockDim.x/2; s>0; s>>=1) 
    {
        if (tid < s) 
        {
            sdata[tid] = mySum = mySum + sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem 
    if (tid == 0) g_odata[blockIdx.x] = sdata[0];
}
#endif

/**
 * Used to set single variables equal to a given value
 *
 * @param g_Var Pointer to global variable
 * @param Value The value to give g_Var
*/
#ifdef __GPU__
__global__
#endif
void setVariable(ValueType *g_Var, ValueType Value) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  
  if (g_Row == 1 && g_Col == 1) {
  #else
  int g_Col = 1;
  int g_Row = 1;
  #endif

  g_Var[0] = Value;

  #ifdef __GPU__
  };
  #endif
};


/**
 * Fixes the velocity gradients so they are smooth
 *
*/
#ifdef __GPU__
__global__
#endif
  void cuFixVelocityGradients(ValueType *g_M, const int UB, const int BB, const int LB, const int RB, const int level) {

  double gradients = 0.0;

  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 

  #else
    for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
  #endif

  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    // UB, BB, LB, RB must be given as the row efter or before the
    // one that is to be filled
    if ( g_Row==UB && UB!=0) { 
      // Halos above
      gradients = -gradVelocity(g_Row,g_Col, g_M, 0, 1, level);
      
      setG( g_Row-1, g_Col, g_M, 
	    gradients*constMem.YNum[level]+getG( g_Row, g_Col, g_M, level) // y = ax+b
	    , level);
    };
    
    if ( g_Row==BB && BB!=0) { 
      // Halos below
      gradients = gradVelocity(g_Row-1,g_Col, g_M, 0, 1, level);
      
      setG( g_Row+1, g_Col, g_M, 
	    gradients*constMem.YNum[level]+getG( g_Row, g_Col, g_M, level) // y = ax+b
	    , level);
    };
    
    if (g_Col==LB && LB!=0) { 
      // Halos left
      gradients = -gradVelocity(g_Row,g_Col, g_M, 1, 0, level);
      
      setG( g_Row, g_Col-1, g_M, 
	    gradients*constMem.XNum[level]+getG( g_Row, g_Col, g_M, level) // y = ax+b
	    , level);
    };
    
    if (g_Col==RB && RB!=0) { 
      // Halos right
      gradients = gradVelocity(g_Row,g_Col-1, g_M, 1, 0, level);
      
      setG( g_Row, g_Col+1, g_M, 
	    gradients*constMem.XNum[level]+getG( g_Row, g_Col, g_M, level) // y = ax+b
	    , level);
    };

  };
  #ifdef __CPU__
      };
    };
#endif
};



/**
 * Used to copy values from one array into another
 *
*/
#ifdef __GPU__
__global__
#endif
  void cuCloneArray(ValueType *g_in, ValueType *g_out, const int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 

  #else
    for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
  #endif


  if ( g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {

    setG(g_Row, g_Col, g_out, getG(g_Row,g_Col, g_in, level), level);

  };
  #ifdef __CPU__
      };
    };
#endif
};
  


















































/*
 * iSOSIA Eqations
 */
#ifdef __iSOSIA__

#ifdef __GPU__               
__global__
#endif
  void ComputeError(const ValueType* u, const ValueType *v, ValueType *error_out, const int level, const int xShift, const int yShift) {

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+xShift;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING+yShift; 

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING+xShift;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING+yShift;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif


	ValueType error = getG(g_Row, g_Col, u, level) - getG(g_Row, g_Col, v, level);

	setG(g_Row, g_Col, error_out, error, level);

      };
#ifdef __CPU__
    }
#endif

};

void SetConstMem(ConstantMem* h_ConstMem) {

  #ifdef __GPU__
    cudaMemcpyToSymbol(constMem, h_ConstMem, sizeof(ConstantMem)); 
  #else
    constMem = (*h_ConstMem);
  #endif
};


/*****
      This function computes the gradients of the ice surface
      and bed surface. Here a possible sediment layer may be 
      included in the bed surface.
      
      Both ice and bed surface must be located in the cell center!
      
      The function returns gradients dhdx, dhdy, dbdx, dbdy.
 */
#ifdef __GPU__               
__global__
#endif
  void cuGetGradients(const ValueType* g_icesurf, 
		      const ValueType *g_bedsurf, 
		      ValueType *g_dhdx, ValueType *g_dhdy, 
		      ValueType *g_dbdx, ValueType *g_dbdy, 
		      int level) {

  double dx = constMem.Dx[level];
  double dy = constMem.Dy[level];

  double maxb = constMem.maxb;
  double maxs = constMem.maxs;

  ValueType dhdx = 0.0;
  ValueType dhdy = 0.0;
  ValueType dbdx = 0.0;
  ValueType dbdy = 0.0;

  #ifdef __GPU__

  // Removing PADDING
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  //g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
  #else
    int g_Col = 0;
    int g_Row = 0;
    for (g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif
	

	if (g_Col > 1 && g_Col < constMem.XNum[level]-1) {
	  //compute gradients in x dir
	  //result will be located on cell wall
	  dbdx = (  getG(g_Row, g_Col  , g_bedsurf, level)
		    - getG(g_Row, g_Col-1, g_bedsurf, level)
		    )/dx;
	  dhdx = (  getG(g_Row, g_Col  , g_icesurf, level)
		    - getG(g_Row, g_Col-1, g_icesurf, level)
		    )/dx;

	} else if ( g_Col == constMem.XNum[level]-1 ) {
	  // On boundaries do a linear extrapolation in x-direction
	  // Using the points (x_k-1, y_k-1) and (x_k, y_k) the point (x*, y(x*))
	  // is computed,
	  //
	  // y(x*) = y_k-1 + (x* - x_k-1 )/( x_k - x_k-1 ) * ( y_k - y_k-1 )
	  //
	  // Read: http://en.wikipedia.org/wiki/Extrapolation

	  // bed(x*)
	  double bstar =
	    getG(g_Row,   constMem.XNum[level]-2, g_bedsurf, level)
	    + ( getG(g_Row,   constMem.XNum[level]-2, g_bedsurf, level) -
		getG(g_Row,   constMem.XNum[level]-3, g_bedsurf, level) );
	  
	  // surf(x*)
	  double istar =
	    getG(g_Row,   constMem.XNum[level]-2, g_icesurf, level)
	    + ( getG(g_Row,   constMem.XNum[level]-2, g_icesurf, level) -
		getG(g_Row,   constMem.XNum[level]-3, g_icesurf, level) );

	  dbdx = ( bstar - getG(g_Row,constMem.XNum[level]-2, g_bedsurf, level)
		   )/dx;
	  dhdx = (  istar - getG(g_Row, constMem.XNum[level]-2, g_icesurf, level)
		    )/dx;

	} else {
	  // On boundaries do a linear extrapolation in x-direction
	  // Using the points (x_k-1, y_k-1) and (x_k, y_k) the point (x*, y(x*))
	  // is computed,
	  //
	  // y(x*) = y_k-1 + (x* - x_k-1 )/( x_k - x_k-1 ) * ( y_k - y_k-1 )
	  //
	  // Read: http://en.wikipedia.org/wiki/Extrapolation

	  // bed(x*)
	  double bstar =
	    getG(g_Row,   1, g_bedsurf, level)
	    -  ( getG(g_Row,   2, g_bedsurf, level) -
		 getG(g_Row,   1, g_bedsurf, level) );
	  
	  // surf(x*)
	  double istar =
	    getG(g_Row,   1, g_icesurf, level)
	    -  ( getG(g_Row,   2, g_icesurf, level) -
		 getG(g_Row,   1, g_icesurf, level) );

	  dbdx = (  getG(g_Row,   1, g_bedsurf, level)
		    - bstar
		    )/dx;
	  dhdx = (  getG(g_Row,   1, g_icesurf, level)
		    - istar
		    )/dx;
	  // dbdx = -10;
	};



	if (g_Row > 1 && g_Row < constMem.YNum[level]-1) {
	  //compute gradients in y directory
	  dbdy = (  getG(g_Row  , g_Col, g_bedsurf, level)
		- getG(g_Row-1, g_Col, g_bedsurf, level)
		    )/dy;
	  dhdy = (  getG(g_Row  , g_Col, g_icesurf, level)
		    - getG(g_Row-1, g_Col, g_icesurf, level)
		    )/dy;

	
	} else if ( g_Row == constMem.YNum[level]-1 ) {
	  // On boundaries do a linear extrapolation in x-direction
	  // Using the points (x_k-1, y_k-1) and (x_k, y_k) the point (x*, y(x*))
	  // is computed,
	  //
	  // y(x*) = y_k-1 + (x* - x_k-1 )/( x_k - x_k-1 ) * ( y_k - y_k-1 )
	  //
	  // Read: http://en.wikipedia.org/wiki/Extrapolation

	  // bed(x*)
	  double bstar =
	    getG(constMem.YNum[level]-2, g_Col, g_bedsurf, level)
	    + ( getG(constMem.YNum[level]-2, g_Col, g_bedsurf, level) -
		getG(constMem.YNum[level]-3, g_Col, g_bedsurf, level) );
	  
	  // surf(x*)
	  double istar =
	    getG(constMem.YNum[level]-2, g_Col, g_icesurf, level)
	    +  ( getG(constMem.YNum[level]-2, g_Col, g_icesurf, level) -
		 getG(constMem.YNum[level]-3, g_Col, g_icesurf, level) );

	  /*
	  dbdy = ( getG(constMem.YNum[level]-3,g_Col, g_bedsurf, level) -  getG(constMem.YNum[level]-4,g_Col, g_bedsurf, level)
		   )/dy;
	  dhdy = (  getG(constMem.YNum[level]-3, g_Col, g_icesurf, level) - getG(constMem.YNum[level]-4, g_Col, g_icesurf, level)
		    )/dy;

*/
	  dbdy = ( bstar - getG(constMem.YNum[level]-2,g_Col, g_bedsurf, level)
		   )/dy;
	  dhdy = (  istar  - getG(constMem.YNum[level]-2, g_Col, g_icesurf, level)
		    )/dy;

	} else  {

	  // bed(x*)
	  double bstar =
	    getG(1,   g_Col, g_bedsurf, level)
	    -  ( getG(2,   g_Col, g_bedsurf, level) -
		 getG(1,   g_Col, g_bedsurf, level) );
	  
	  // surf(x*)
	  double istar =
	    getG(1,   g_Col, g_icesurf, level)
	    - ( getG(2,   g_Col, g_icesurf, level) -
		getG(1,   g_Col, g_icesurf, level) );

	  dbdy = (  getG(1,   g_Col, g_bedsurf, level)
		    - bstar
		    )/dy;
	  dhdy = (  getG(1,   g_Col, g_icesurf, level)
		    - istar
		    )/dy;
	  /*
	  dbdy = (  getG(2,   g_Col, g_bedsurf, level)
		    - getG(1,   g_Col, g_bedsurf, level)
		    )/dy;
	  dhdy = (  getG(2,   g_Col, g_icesurf, level)
		    - getG(1,   g_Col, g_icesurf, level)
		    )/dy;
	  */
	}
	
	if (dbdx < -maxb) dbdx = -maxb;
	if (dbdx > maxb)  dbdx = maxb;
	if (dhdx < -maxs) dhdx = -maxs;
	if (dhdx > maxs)  dhdx = maxs;
	
	if (dbdy < -maxb) dbdy = -maxb;
	if (dbdy > maxb)  dbdy = maxb;
	if (dhdy < -maxs) dhdy = -maxs;
	if (dhdy > maxs)  dhdy = maxs;

	// Save values to global memory

	setG(g_Row, g_Col, g_dhdx, dhdx, level);
	setG(g_Row, g_Col, g_dhdy, dhdy, level);
	setG(g_Row, g_Col, g_dbdx, dbdx, level);
	setG(g_Row, g_Col, g_dbdy, dbdy, level);

  #ifdef __CPU__
      } 
  #endif
    }

  }

#ifdef __GPU__               
__global__
#endif
  void cuInterpGradients(const ValueType *g_dhdx, const ValueType *g_dhdy, 
			 const ValueType *g_dbdx, const ValueType *g_dbdy,
			 ValueType *g_dhdx_c, ValueType *g_dhdy_c, 
			 ValueType *g_dbdx_c, ValueType *g_dbdy_c, 
			 int level) {

  ValueType dhdx = 0.0;
  ValueType dhdy = 0.0;
  ValueType dbdx = 0.0;
  ValueType dbdy = 0.0;

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x;//+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y;//+PADDING; 
  //g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    int g_Col = 0;
    int g_Row = 0;
    for (g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

	dhdx = (getG(g_Row, g_Col  , g_dhdx, level) + 
		getG(g_Row, g_Col+1, g_dhdx, level))/2.0;
	dbdx = (getG(g_Row, g_Col  , g_dbdx, level) + 
		getG(g_Row, g_Col+1, g_dbdx, level))/2.0;
	dhdy = (getG(g_Row,   g_Col, g_dhdy, level) + 
		getG(g_Row+1, g_Col, g_dhdy, level))/2.0;
	dbdy = (getG(g_Row,   g_Col, g_dbdy, level) + 
		getG(g_Row+1, g_Col, g_dbdy, level))/2.0;

	setG(g_Row, g_Col, g_dhdx_c, dhdx, level);
	setG(g_Row, g_Col, g_dbdx_c, dbdx, level);
	setG(g_Row, g_Col, g_dhdy_c, dhdy, level);
	setG(g_Row, g_Col, g_dbdy_c, dbdy, level);

  #ifdef __CPU__
      } 
  #endif
    }

  }

#ifdef __GPU__               
__global__
#endif
void VerifyConstMem(ConstantMem* d_ConstMem, int* Error) {

  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  #else
  int g_Col = 1;
  int g_Row = 1;

  #endif

  // Only Allow one thread!
  if (g_Col == 1 && g_Row == 1) {
  // Check XNum and YNum

    (*Error) = 0; // No errors yet

    if (d_ConstMem->LevelNum == 0 ||
	d_ConstMem->LevelNum > maxLevel){
      (*Error) = -10;
      return;
    };

    // Check arrays
    for (int i=0; i<d_ConstMem->LevelNum;i++){
      if(d_ConstMem->XNum[i] != constMem.XNum[i]  ||
	 d_ConstMem->YNum[i] != constMem.YNum[i]  ||
 	 d_ConstMem->Dx[i] != constMem.Dx[i]      ||
	 d_ConstMem->Dy[i] != constMem.Dy[i]      ||
	 d_ConstMem->IterNum[i] != constMem.IterNum[i])  {
	(*Error) = -11;
	return;
      };
    };
    
    // Check BC 
    if (d_ConstMem->BC[0] != constMem.BC[0] || 
	d_ConstMem->BC[1] != constMem.BC[1] || 
	d_ConstMem->BC[2] != constMem.BC[2] || 
	d_ConstMem->BC[3] != constMem.BC[3] ) {
      (*Error) = -11;
      return;
    };

    // Check single variables
    if (d_ConstMem->XDim != constMem.XDim     ||
	d_ConstMem->YDim != constMem.YDim     ||
	d_ConstMem->Relaxs != constMem.Relaxs ||
	d_ConstMem->Relaxv != constMem.Relaxv ||
	d_ConstMem->Gamma != constMem.Gamma   ||
	d_ConstMem->A != constMem.A           ||
	d_ConstMem->vKoef != constMem.vKoef   ||
	d_ConstMem->sKoef != constMem.sKoef   ||
	d_ConstMem->LevelNum != constMem.LevelNum) {
      (*Error) = -12;
      return;
    };

  };
};


/**
 * Update the velocity in the x/x_1-direction matrix based on current gradients
 * 
 * Note: If ComputeSurfaceVelocity is eq 1 then only the approksimation 1.2*Vy 
 * is computed.
 *
 * Note: The coefficients are not saved outside this function!
 *
 * @param g_Current The current value computed from the last iteration
 * @param g_Suggested The new suggested value from the coefficients
 * @param g_Wanted The current solution to solve for. This value will never be updated in this function
 * @param g_Error Current error/difference between suggested and current value
 * @param g_Vxs Pointer to surface velocity
 * @param g_Sxx Global pointer to Sxx stress matrix
 * @param g_Syy Global pointer to Syy stress matrix
 * @param g_Sxy Global pointer to Sxy stress matrix
 * @param g_Sxym Global pointer to Sxy stress matrix interpolated to corner nodes
 * @param g_IceTopo Global pointer to ice topography matrix
 * @param g_IceTopom Global pointer to interpolated ice topography values in corner nodes
 * @param g_BedTopo Global pointer to bed topography matrix
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
 * @param ComputeSurfaceVelocity Compute the surface velocity
 * @param UpdateWanted Update the wanted matrix if first run on new level
*/
#ifdef __GPU__               
__global__
#endif
  void g_update_vx(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, 
		   ValueType *g_Vx, ValueType *g_Vxs, ValueType *g_Vxb, ValueType *g_Ezz,
		   const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
		   const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
		   const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
		   const ValueType *g_dhdx, const ValueType *g_dhdy,
		   const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted,
		   const bool PostProlongUpdate, const int Blockwise = 0)   
{

  double maxdef = 1500;

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+Blockwise;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
       	if ((g_Row*g_Col % 2) == color) {
  #endif


	ValueType H = (
		       (getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level))
		       + (getG(g_Row, g_Col-1, g_IceTopo, level) - getG(g_Row, g_Col-1, g_BedTopo, level))
		       )/2.0; 
	
	if ( g_Col > 1 && H > 5.0 ) { 

	// Interpolate stresses to velocity node position
	ValueType Sxx = ((getG(g_Row, g_Col, g_Sxx, level)+getG(g_Row, g_Col-1, g_Sxx, level))/2.0);
	ValueType Syy = ((getG(g_Row, g_Col, g_Syy, level)+getG(g_Row, g_Col-1, g_Syy, level))/2.0);
	ValueType Sxy = ((getG(g_Row, g_Col, g_Sxy, level)+getG(g_Row, g_Col-1, g_Sxy, level))/2.0);

	ValueType dhdx = getG(g_Row, g_Col, g_dhdx, level);
	ValueType dhdy = (getG(g_Row, g_Col, g_dhdy, level)+getG(g_Row, g_Col-1, g_dhdy, level))/2.0;

	ValueType dsxxdy = (grad(g_Row, g_Col, g_Sxx, 0, 1, level)+grad(g_Row, g_Col-1, g_Sxx, 0, 1, level))/2.0;
	ValueType dsyydy = (grad(g_Row, g_Col, g_Syy, 0, 1, level)+grad(g_Row, g_Col-1, g_Syy, 0, 1, level))/2.0;
	ValueType dsxydy = (grad(g_Row, g_Col, g_Sxy, 0, 1, level)+grad(g_Row, g_Col-1, g_Sxy, 0, 1, level))/2.0;

	/*
	ValueType dsxxdy = gradVelocity(g_Row, g_Col, g_Sxxm, 0, 1, level);
	ValueType dsyydy = gradVelocity(g_Row, g_Col, g_Syym, 0, 1, level);
	ValueType dsxydy = gradVelocity(g_Row, g_Col, g_Sxym, 0, 1, level);
	*/
	ValueType alpha = dhdx*dhdx + dhdy*dhdy;

	ValueType cx0 = (2.0 * Sxx + Syy)* dhdx + Sxy * dhdy;
	
	ValueType cx1 = 
	  - (1.0 + alpha) * dhdx
	  + 2.0 * gradVelocity(g_Row  , g_Col-1, g_Sxx, 1, 0, level)  // d(Sxx)/dx
	  +       gradVelocity(g_Row  , g_Col-1, g_Syy, 1, 0, level)  // d(Syy)/dx
	  +       dsxydy;
	
	// Compute y-cofficients in velocity position
	ValueType cy0 = (Sxx + 2.0 * Syy) * dhdy + Sxy * dhdx;
	
	ValueType cy1 = 
	  - (1.0 + alpha) * dhdy
	  +       dsxxdy
	  + 2.0 * dsyydy
	  +       gradVelocity(g_Row  , g_Col-1, g_Sxy, 1, 0, level); // d(Sxy)/dx

	ValueType taud2 =  Sxx*Sxx + Syy*Syy + Sxx*Syy + Sxy*Sxy;
	
	ValueType k0 = cx0*cx0 + cy0*cy0 + taud2;
	ValueType k1 = 2.0 * (cx0*cx1 + cy0*cy1);
	ValueType k2 = cx1*cx1 + cy1*cy1;
		
	// x-coefficients
	ValueType wx1 = (cx0 * k0);/* - ((getG(g_Row, g_Col, g_Vxb, level) - getG(g_Row-1, g_Col, g_Vxb, level))/constMem.Dx[level]  // dvxdx
				      -
				      (getG(g_Row, g_Col, g_BedTopo, level) - getG(g_Row-1, g_Col, g_BedTopo, level))/constMem.Dx[level] * // dbdx
				      (getG(g_Row, g_Col, g_Ezz, level)+getG(g_Row-1, g_Col, g_Ezz, level) 
				      )/2.0)/(2.0 * constMem.A); // The zero is from d(vz_b)/dx  */
	ValueType wx2 = (cx0*k1 + cx1*k0);/* - 1.0/(4.0*constMem.A)*(gradVelocity(g_Row, g_Col, g_Ezz, 1, 0, level)); */
	ValueType wx3 = (cx0*k2 + cx1*k1);
	ValueType wx4 = (cx1*k2);
	
	if (UpdateWanted) {
	  ValueType Vx_suggested = 2.0 * constMem.A * (wx1*H/2.0 + wx2*H*H/3.0 + wx3*H*H*H/4.0 + wx4*H*H*H*H/5.0);
	  setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( Vx_suggested - getG(g_Row, g_Col, g_Current, level)), level);
	  
	} else {
	  
	  if ( ComputeSurfaceVelocity ) {

	    ValueType Vx_Surface = 0.0;

	    // Compute approximation or full term?
	    if  ( ComputeSurfaceVelocity == 1 ) {
	      Vx_Surface = 1.2*getG(g_Row, g_Col, g_Current, level);
	    } else {
	      Vx_Surface = 2.0 * constMem.A * (wx1*H + wx2*H*H/2.0 + wx3*H*H*H/3.0 + wx4*H*H*H*H/4.0);
	    };

	    setG(g_Row, g_Col, g_Vxs, Vx_Surface, level);
	  } else {
	    ValueType Vx_suggested = 2.0 * constMem.A * 
	      (wx1*H/2.0 + wx2*H*H/3.0 + wx3*H*H*H/4.0 + wx4*H*H*H*H/5.0);
	    if (fabs(Vx_suggested) > maxdef) Vx_suggested *= maxdef/fabs(Vx_suggested);
	    ValueType Vx_error     = (Vx_suggested - getG(g_Row, g_Col, g_Current, level));
	    ValueType Vx_new = getG(g_Row, g_Col, g_Current, level) + Vx_error*constMem.Relaxv;

	    setG(g_Row, g_Col, g_Current  , Vx_new      , level); 
	    setG(g_Row, g_Col, g_Suggested, Vx_suggested, level);
	    setG(g_Row, g_Col, g_Error    , Vx_error    , level);
	  } 
	};  // END OF ELSE (UpdateWanted)	
	} else {
	  setG(g_Row, g_Col, g_Current  , 0.0 , level); 
	  setG(g_Row, g_Col, g_Error    , 0.0 , level);
	  setG(g_Row, g_Col, g_Suggested    , 0.0 , level);
	}; // END OF ELSE (g_Col > 1)
	};
#ifdef __CPU__
       }
    }
  #endif
  };






/**
 * Update the velocity in the y/x_2-direction matrix based on current gradients
 *
 * Note: If ComputeSurfaceVelocity is eq 1 then only the approksimation 1.2*Vy 
 * is computed.
 * 
 * Note: The coefficients are not saved outside this function!
 *
 * @param g_Current The current value computed from the last iteration
 * @param g_Suggested The new suggested value from the coefficients
 * @param g_Wanted The current solution to solve for. This value will never be updated in this function
 * @param g_Error Current error/difference between suggested and current value
 * @param g_Sxx Global pointer to Sxx stress matrix
 * @param g_Syy Global pointer to Syy stress matrix
 * @param g_Sxy Global pointer to Sxy stress matrix
 * @param g_Sxym Global pointer to Sxy stress matrix in corner nodes
 * @param g_IceTopo Global pointer to ice topography matrix
 * @param g_IceTopom Global pointer to interpolated ice topography values in corner nodes
 * @param g_BedTopo Global pointer to bed topography matrix
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
 * @param ComputeSurfaceVelocity Compute the surface velocity
 * @param UpdateWanted Update the wanted matrix if first run on new level
*/
#ifdef __GPU__               
__global__
#endif
  void g_update_vy(ValueType *g_Current, ValueType *g_Suggested,  ValueType *g_Wanted, ValueType *g_Error,
		   ValueType *g_Vy, ValueType *g_Vys, ValueType *g_Vyb, ValueType *g_Ezz,
		   const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
		   const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
		   const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
		   const ValueType *g_dhdx, const ValueType *g_dhdy,
		   const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted,
		   const bool PostProlongUpdate, const int Blockwise = 0)   
{

  double maxdef = 1500;

  #ifdef __GPU__  
  int g_Col = (blockIdx.x*blockDim.x+threadIdx.x+PADDING); // Thread Col
  int g_Row = (blockIdx.y*blockDim.y+threadIdx.y+PADDING); // Thread Row
  g_Row = ((color+g_Col)%2)+2*g_Row-1+Blockwise;

  
  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
     
  for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
       	if ((g_Row*g_Col % 2) == color) {
  #endif

	    ValueType H = (
			   (getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level))
			   + (getG(g_Row-1, g_Col, g_IceTopo, level) - getG(g_Row-1, g_Col, g_BedTopo, level))
			   )/2.0; 
				    
       	if ( g_Row > 1 && H > 5.0 ) {
	   
	    // Compute x-cofficients in velocity position
	    ValueType Sxx = ((getG(g_Row, g_Col, g_Sxx, level)+getG(g_Row-1, g_Col, g_Sxx, level))/2.0);
	    ValueType Syy = ((getG(g_Row, g_Col, g_Syy, level)+getG(g_Row-1, g_Col, g_Syy, level))/2.0);
	    ValueType Sxy = ((getG(g_Row, g_Col, g_Sxy, level)+getG(g_Row-1, g_Col, g_Sxy, level))/2.0);

	    ValueType dhdx = (getG(g_Row, g_Col, g_dhdx, level)+getG(g_Row-1, g_Col, g_dhdx, level))/2.0;//gradVelocity(g_Row, g_Col-1, g_IceTopo  , 1, 0, level);
	    ValueType dhdy = getG(g_Row, g_Col, g_dhdy, level);//+getG(g_Row-1, g_Col, g_dhdy, level))/2.0;//gradVelocity(g_Row, g_Col  , g_IceTopom , 0, 1, level);

	    ValueType dsxxdx = (grad(g_Row, g_Col, g_Sxx, 1, 0, level)+grad(g_Row-1, g_Col, g_Sxx, 1, 0, level))/2.0;
	    ValueType dsyydx = (grad(g_Row, g_Col, g_Syy, 1, 0, level)+grad(g_Row-1, g_Col, g_Syy, 1, 0, level))/2.0;
	    ValueType dsxydx = (grad(g_Row, g_Col, g_Sxy, 1, 0, level)+grad(g_Row-1, g_Col, g_Sxy, 1, 0, level))/2.0;
	    

	    /*	        
	    ValueType dsxxdx = gradVelocity(g_Row, g_Col, g_Sxxm, 1, 0, level);
	    ValueType dsyydx = gradVelocity(g_Row, g_Col, g_Syym, 1, 0, level);
	    ValueType dsxydx = gradVelocity(g_Row, g_Col, g_Sxym, 1, 0, level);
	    */

	    ValueType alpha = dhdx*dhdx + dhdy*dhdy;

	    ValueType cx0 = (2.0 * Sxx + Syy)* dhdx + Sxy * dhdy;
	
	    ValueType cx1 = 
	      - (1.0 + alpha) * dhdx
	      +  2.0 * dsxxdx
	      +        dsyydx
	      +        gradVelocity(g_Row-1, g_Col  , g_Sxy     , 0, 1, level); // d(Sxy)/dy
	    
	    // Compute y-cofficients in velocity position
	    ValueType cy0 = (Sxx + 2.0 * Syy) * dhdy + Sxy * dhdx;
	    
	    ValueType cy1 = 
	      - (1.0 + alpha) * dhdy
	      +       gradVelocity(g_Row-1, g_Col, g_Sxx    , 0, 1, level)   // d(Sxx)/dy
	      + 2.0 * gradVelocity(g_Row-1, g_Col, g_Syy    , 0, 1, level)   // d(Syy)/dy
	      +       dsxydx;
	    
	    ValueType taud2 = Sxx*Sxx + Syy*Syy + Sxx*Syy + Sxy*Sxy;
	    
	    ValueType k0 = cx0*cx0 + cy0*cy0 + taud2;
	    ValueType k1 = 2.0 * (cx0*cx1 + cy0*cy1);
	    ValueType k2 = cx1*cx1 + cy1*cy1;
	    
	      // y-coefficients
	    ValueType wy1 = (cy0 * k0);/* - ((getG(g_Row, g_Col, g_Vyb, level) - getG(g_Row, g_Col-1, g_Vyb, level))/constMem.Dy[level] - 
					 (getG(g_Row, g_Col, g_BedTopo, level) - getG(g_Row, g_Col-1, g_BedTopo, level))/constMem.Dy[level] *
				         (getG(g_Row, g_Col, g_Ezz, level)+getG(g_Row, g_Col-1, g_Ezz, level) 
				         )/2.0)/(2.0 * constMem.A); // The zero is from d(vz_b)/dy*/
	    ValueType wy2 = (cy0*k1 + cy1*k0);// - 1.0/(4.0*constMem.A)*(gradVelocity(g_Row, g_Col, g_Ezz, 0, 1, level));
	    ValueType wy3 = (cy0*k2 + cy1*k1);
	    ValueType wy4 = (cy1*k2);

	      if (UpdateWanted) {
		ValueType f = getG(g_Row, g_Col, g_Wanted, level) 
		  + ( (2.0 * constMem.A * (wy1*H/2.0 + wy2*H*H/3.0 + wy3*H*H*H/4.0 + wy4*H*H*H*H/5.0) - getG(g_Row, g_Col, g_Current, level)) 
		      );

		setG(g_Row, g_Col, g_Wanted, f, level);
		
	      } else {


	      if ( ComputeSurfaceVelocity ) {
		
		ValueType Vy_Surface = 0.0;

		// Compute approximation or full term?
		if  ( ComputeSurfaceVelocity == 1 ) {
		  Vy_Surface = 1.2*getG(g_Row, g_Col, g_Current, level);
		} else {
		  Vy_Surface = 2.0 * constMem.A * (wy1*H + wy2*H*H/2.0 + wy3*H*H*H/3.0 + wy4*H*H*H*H/4.0);
		};

		setG(g_Row, g_Col, g_Vys, Vy_Surface, level);
	      } else {
		
		  ValueType Vy_suggested = 2.0 * constMem.A * (wy1*H/2.0 + wy2*H*H/3.0 + wy3*H*H*H/4.0 + wy4*H*H*H*H/5.0);
		  if (fabs(Vy_suggested) > maxdef) Vy_suggested *= maxdef/fabs(Vy_suggested);
		  ValueType Vy_error     = ( Vy_suggested - getG(g_Row, g_Col, g_Current, level) );
		  ValueType Vy_new = getG(g_Row, g_Col, g_Current, level) + Vy_error*constMem.Relaxv;

		  setG(g_Row, g_Col, g_Current  , Vy_new      , level);
		  setG(g_Row, g_Col, g_Suggested, Vy_suggested, level);
		  setG(g_Row, g_Col, g_Error    , Vy_error    , level);			      	      
	      }
	      }
	} else {
	  setG(g_Row, g_Col, g_Current  , 0.0 , level);
	  setG(g_Row, g_Col, g_Error    , 0.0 , level);
	  setG(g_Row, g_Col, g_Suggested    , 0.0 , level);		
	};  // END OF ELSE ( g_Row > 2) 
	};
	
#ifdef __CPU__
       }
    }
  #endif
};


/**
 * Computes the terms used for sliding. The update of the current sliding speed is done in
 * g_update_sliding!
 *
 * Values are computed in the center of each node.
 *
 * @param g_Current The current value computed from the last iteration
 * @param g_Suggested The new suggested value from the coefficients
 * @param g_Wanted The current solution to solve for. This value will never be updated in this function
 * @param g_Error Current error/difference between suggested and current value
 * @param g_Sxx Global pointer to Sxx stress matrix
 * @param g_Syy Global pointer to Syy stress matrix
 * @param g_Sxy Global pointer to Sxy stress matrix
 * @param g_Sxym Global pointer to Sxy stress matrix in corner nodes
 * @param g_IceTopo Global pointer to ice topography matrix
 * @param g_IceTopom Global pointer to interpolated ice topography values in corner nodes
 * @param g_BedTopo Global pointer to bed topography matrix
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
 * @param ComputeSurfaceVelocity Compute the surface velocity
 * @param UpdateWanted Update the wanted matrix if first run on new level
*/
#ifdef __GPU__               
__global__
#endif
  void g_compute_sliding_terms(ValueType *g_Tbx, ValueType *g_Tby, ValueType *g_Tbz, ValueType *g_Vb, ValueType *g_Ts, ValueType *g_Tn, ValueType *g_beta,
			       const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy,
			       const ValueType *g_Pw, const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
			       const ValueType *g_dhdx, const ValueType *g_dhdy, const ValueType *g_dbdx, const ValueType *g_dbdy,
			       const int level, const int color)   
{
  
  #ifdef __GPU__  
  int g_Col = (blockIdx.x*blockDim.x+threadIdx.x+PADDING); // Thread Col
  int g_Row = (blockIdx.y*blockDim.y+threadIdx.y+PADDING); // Thread Row
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
     
  for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {

       	if ((g_Row*g_Col % 2) == color) {
  #endif


	  ValueType dhdx = getG(g_Row, g_Col, g_dhdx, level);
	  ValueType dhdy = getG(g_Row, g_Col, g_dhdy, level);	  

	  ValueType alpha = pow(dhdx,2) + pow(dhdy,2); 	  // dhdx^2 + dhdy^2

	  ValueType cx0 = 
	    (2.0 * getG(g_Row, g_Col, g_Sxx, level) 
	         + getG(g_Row, g_Col, g_Syy, level)
	     ) * dhdx 
	    + getG(g_Row, g_Col, g_Sxy, level) * dhdy;

	  ValueType cx1 = 
	    - (1.0 + alpha) * dhdx
	    + 2.0 * grad(g_Row, g_Col, g_Sxx    , 1, 0, level)   // d(Sxx)/dx
	    +       grad(g_Row, g_Col, g_Syy    , 1, 0, level)   // d(Syy)/dx
	    +       grad(g_Row, g_Col, g_Sxy    , 0, 1, level);  // d(Sxy)/dy

	  // Compute y-cofficients
	  ValueType cy0 = 
	    (        getG(g_Row, g_Col, g_Sxx, level) 
	     + 2.0 * getG(g_Row, g_Col, g_Syy, level)
		     ) * dhdy 
	    + getG(g_Row, g_Col, g_Sxy, level) * dhdx;

	  ValueType cy1 = 
	    - (1.0 + alpha) * dhdy
	    +       grad(g_Row, g_Col, g_Sxx    , 0, 1, level)   // d(Sxx)/dy
	    + 2.0 * grad(g_Row, g_Col, g_Syy    , 0, 1, level)   // d(Syy)/dy
	    +       grad(g_Row, g_Col, g_Sxy    , 1, 0, level);  // d(Sxy)/dx

	  ValueType H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); // Ice thickness
	  
	  ValueType dbdx = getG(g_Row, g_Col, g_dbdx, level);
	  ValueType dbdy = getG(g_Row, g_Col, g_dbdy, level);

	  ValueType szz = 0.0;
	  if (H > 5.0) {
	    szz = -(getG(g_Row, g_Col, g_Sxx, level) + getG(g_Row, g_Col, g_Syy, level));
	  }
	  
	  ValueType lb = sqrt(1.0+pow(dbdx,2.0)+pow(dbdy,2.0)); 

	  ValueType pb = (1.0+pow(dbdx,2.0)+pow(dbdy,2.0))*H+szz; 	  // Pressure at the bed
	  ValueType sxz = cx0+cx1*H;
	  ValueType syz = cy0+cy1*H;

	  // Calculate basal normal traction
	  ValueType tn = pb + ( 
			       2.0 * dbdx * sxz
			       + 2.0 * dbdy * syz
			       - pow(dbdx,2.0)
			       * getG(g_Row, g_Col, g_Sxx, level) - pow(dbdy,2.0)
			       * getG(g_Row, g_Col, g_Syy, level) - szz
			       - 2.0 * dbdx * dbdy * getG(g_Row, g_Col, g_Sxy, level)
				)/pow(lb,2.0);
	  if ( tn < 0.0) tn = 0.0;

	  // Effective pressure
	  ValueType te = tn-getG(g_Row, g_Col, g_Pw, level);///(constMem.g*constMem.rho_w); 
	  if (te < 1.0) te = 1.0;

	  // Basal traction components
	  ValueType tbx = (
			   (1.0-pow(dbdx,2.0)+pow(dbdy,2.0))
			   * sxz-2.0*dbdx*dbdy*syz
			   - (1.0+pow(dbdy,2.0))
			   * dbdx*getG(g_Row, g_Col, g_Sxx, level)+dbdx*pow(dbdy,2.0)
			   * getG(g_Row, g_Col, g_Syy, level)+dbdx*szz
			   - (1.0-pow(dbdx,2.0)+pow(dbdy,2.0))
			   * dbdy*getG(g_Row, g_Col, g_Sxy, level)
			   )/pow(lb,3.0);

	  ValueType tby = (-2.0*dbdx*dbdy*sxz
			     + (1.0+pow(dbdx,2.0)-pow(dbdy,2.0))
			   * syz+pow(dbdx,2.0)*dbdy*getG(g_Row, g_Col, g_Sxx, level)-(1.0+pow(dbdx,2.0))
			   * dbdy*getG(g_Row, g_Col, g_Syy, level)+dbdy*szz-(1.0+pow(dbdx,2.0)-pow(dbdy,2.0))
			   * dbdx*getG(g_Row, g_Col, g_Sxy, level)
			   )/pow(lb,3.0);

	  ValueType tbz = (
			   (1.0-pow(dbdx,2.0)-pow(dbdy,2.0))
			   * dbdx*sxz+(1.0-pow(dbdx,2.0)-pow(dbdy,2.0))
			   * dbdy*syz-pow(dbdx,2.0)*getG(g_Row, g_Col, g_Sxx, level)
			   - pow(dbdy,2.0)*getG(g_Row, g_Col, g_Syy, level)
			   + (pow(dbdx,2.0)+pow(dbdy,2.0))
			   * szz-2.0*dbdx*dbdy*getG(g_Row, g_Col, g_Sxy, level)
			   )/pow(lb,3.0);

	  // basal shear traction
	  ValueType ts = sqrt(pow(tbx,2.0)+pow(tby,2.0)+pow(tbz,2.0));

	  // bed velocity
	  ValueType vb = 0.0;
	  
	  // Weertman sliding
	  if (constMem.slidingmode == 0) {
	    if (H > 10.0) vb = constMem.Cs*ts*ts;//(te+10.0);
	    else if (H > 1.0) vb = (constMem.Cs*H/10.0)*ts*ts;//(te+10.0);
	    else vb = 0.0;
	    
	    if (vb > constMem.maxsliding) vb = constMem.maxsliding;
	  } else {
	    
	    // Schoof sliding	  	  
	    ValueType beta = pow(constMem.C,3.0)-pow(ts/(te+1.0),3.0);
	    if (beta < constMem.minbeta) beta = constMem.minbeta;
	    if (beta <= constMem.L0*pow(ts,3.0)/constMem.maxsliding) vb = constMem.maxsliding;
	    else if (H > 5.0) vb = constMem.L0*pow(ts,3.0)/beta;
	    else if (H > 1.0) vb = (constMem.L0/H)*pow(ts,3.0)/beta;
	    else vb = 0.0;	           

	  // Save beta
	  setG(g_Row, g_Col, g_beta, beta, level);	  

	  }

	  // Save basal traction
	  setG(g_Row, g_Col, g_Tbx, tbx, level);
	  setG(g_Row, g_Col, g_Tby, tby, level);
	  setG(g_Row, g_Col, g_Tbz, tbz, level);

	  // Save bed velocity
	  if (constMem.dosliding == 1)
	    setG(g_Row, g_Col, g_Vb, vb, level);
	  else
	    setG(g_Row, g_Col, g_Vb, 0.0, level);

	  // basal shear traction
	  setG(g_Row, g_Col, g_Ts, ts, level);
	  setG(g_Row, g_Col, g_Tn, tn, level);

        };
  #ifdef __CPU__
       }
    }
  #endif
};

/**
 * Update sliding in x-direction based on coefficients in g_compute_sliding_terms
 *
*/

#ifdef __GPU__               
__global__
#endif
  void g_update_vxb(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
		    ValueType *g_Vxd, ValueType *g_Vx,
		    const ValueType *g_Tbx,
		    const ValueType *g_Ts, const ValueType *g_Vb, 
		    const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
		    const int level, const int color, const int Blockwise = 0)   
{

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+Blockwise;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else

    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif
       	if ( g_Col > 1 ) {

	   // Interpolate values to velocity cells
	   ValueType tbx = 0.5*(getG(g_Row, g_Col-1, g_Tbx, level)+getG(g_Row, g_Col, g_Tbx, level));
	   ValueType ts  = 0.5*(getG(g_Row, g_Col-1, g_Ts , level)+getG(g_Row, g_Col, g_Ts , level));
	   ValueType vb  = 0.5*(getG(g_Row, g_Col-1, g_Vb , level)+getG(g_Row, g_Col, g_Vb , level));
	   
	   // Compute update
	   ValueType H = (
			  (getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level))
			  + (getG(g_Row, g_Col-1, g_IceTopo, level) - getG(g_Row, g_Col-1, g_BedTopo, level))
			  )/2.0f; 

	   if (H > 1.0) {
	     ValueType Vxb_suggested = vb*tbx/(ts+1.0e-6);
	     //	     ValueType Vxb_error     = getG(g_Row, g_Col, g_Wanted, level) - ( Vy_suggested - getG(g_Row, g_Col, g_Current, level));
	     ValueType Vxb_new       = (1.0 - constMem.Relaxvb)*getG(g_Row, g_Col, g_Current, level) + constMem.Relaxvb*Vxb_suggested;

	     // Update values with new sliding velocities
	     setG(g_Row, g_Col, g_Current  , Vxb_new      , level);	     
	     setG(g_Row, g_Col, g_Suggested, Vxb_suggested, level);
	     setG(g_Row, g_Col, g_Vx, getG(g_Row, g_Col, g_Vxd, level) + Vxb_new, level);
	     
	     //	     setG(g_Row, g_Col, g_Suggested, Vy_suggested, level);
	     //	     setG(g_Row, g_Col, g_Error    , Vy_error    , level);
	   } else {
	     setG(g_Row, g_Col, g_Current, 0.0, level);
	   }
	   };
	 };
#ifdef __CPU__
	};	   
    }
#endif
};

/**
 * Update sliding in y-direction based on coefficients in g_compute_sliding_terms
 *
*/

#ifdef __GPU__               
__global__
#endif
  void g_update_vyb(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
		    ValueType *g_Vyd, ValueType *g_Vy,
		    const ValueType *g_Tby,
		    const ValueType *g_Ts, const ValueType *g_Vb, 
		    const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
		    const int level, const int color, const int Blockwise = 0)   
{

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1+Blockwise;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else

    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

       	if ( g_Row > 1 ) {

	   // Interpolate values to velocity cells
	   ValueType tby = 0.5*(getG(g_Row-1, g_Col, g_Tby, level)+getG(g_Row, g_Col, g_Tby, level));
	   ValueType ts  = 0.5*(getG(g_Row-1, g_Col, g_Ts , level)+getG(g_Row, g_Col, g_Ts , level));
	   ValueType vb  = 0.5*(getG(g_Row-1, g_Col, g_Vb , level)+getG(g_Row, g_Col, g_Vb , level));
	   
	   // Compute update
	    ValueType H = (
			   (getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level))
			   + (getG(g_Row-1, g_Col, g_IceTopo, level) - getG(g_Row-1, g_Col, g_BedTopo, level))
			   )/2.0f; 

	    if (H > 1.0) {
	      ValueType Vyb_suggested = vb*tby/(ts+1.0e-6);
	      //	     ValueType Vxb_error     = getG(g_Row, g_Col, g_Wanted, level) - ( Vy_suggested - getG(g_Row, g_Col, g_Current, level));
	      ValueType Vyb_new       = (1.0 - constMem.Relaxvb)*getG(g_Row, g_Col, g_Current, level) + constMem.Relaxvb*Vyb_suggested;
	      //	      ValueType Vyb_new = 0.0;
	     // Update velocities
	     setG(g_Row, g_Col, g_Current  , Vyb_new      , level);
	     setG(g_Row, g_Col, g_Vy, getG(g_Row, g_Col, g_Vyd, level) + Vyb_new, level);
	     setG(g_Row, g_Col, g_Suggested, Vyb_suggested, level);
	     //	     setG(g_Row, g_Col, g_Error    , Vy_error    , level);
	    } else {
	      setG(g_Row, g_Col, g_Current, 0.0, level);
	    }

	   };
	 };
#ifdef __CPU__
	};	   
    }
#endif
};
	

  /**
   * Computes the error in sliding estimates.
   *
   * This value is computed in the cell center.
   *
   **/
#ifdef __GPU__               
__global__
#endif
  void g_Compute_sliding_residual(ValueType *g_VbRes,
				  const ValueType *g_Vbx, const ValueType *g_Vby, const ValueType *g_Vb, 
				  const int level)   
{

  int g_Col, g_Row;
  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  //  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	  //	 if ((g_Row*g_Col % 2) == color) {
  #endif

	  ValueType vbx  = 0.5*(getG(g_Row, g_Col, g_Vbx , level)+getG(g_Row,   g_Col+1, g_Vbx , level));
	  ValueType vby  = 0.5*(getG(g_Row, g_Col, g_Vby , level)+getG(g_Row+1, g_Col,   g_Vby , level));
	  ValueType sliding = sqrt(vbx*vbx + vby*vby);
	  
	  // Residual
	  setG(g_Row, g_Col, g_VbRes, (getG(g_Row, g_Col, g_Vb, level) - sliding), level);

  #ifdef __CPU__
        }
  #endif
  }
}

  /**
 * Update the effective stress (Tau_e) matrix based on current gradients
 *
 * Note: The coefficients are not saved outside this function!
 *
 * @param g_Current The current value computed from the last iteration
 * @param g_Suggested The new suggested value from the coefficients
 * @param g_Wanted The current solution to solve for. This value will never be updated in this function
 * @param g_Error Current error/difference between suggested and current value
 * @param g_Sxx Global pointer to Sxx stress matrix
 * @param g_Syy Global pointer to Syy stress matrix
 * @param g_Sxy Global pointer to Sxy stress matrix
 * @param g_IceTopo Global pointer to ice topography matrix
 * @param g_BedTopo Global pointer to bed topography matrix
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
*/
#ifdef __GPU__               
__global__
#endif
  void g_update_effective_stress(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
				 const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
				 const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
				 const ValueType *g_dhdx, const ValueType *g_dhdy,
				 const int level, const int color, const bool UpdateWanted)   
{

  ValueType dhdx, dhdy, alpha, cx0, cx1, cy0, cy1;
  ValueType taud2 = 0, k0, k1, k2, H;
  ValueType Taue_new, Taue_suggested, Taue_error;
  int g_Col, g_Row;

  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

	   H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); 
	   dhdx = getG(g_Row, g_Col, g_dhdx, level);
	   dhdy = getG(g_Row, g_Col, g_dhdy, level);	  

	   alpha = dhdx*dhdx + dhdy*dhdy; 	  // dhdx^2 + dhdy^2
	   
	  // Compute x-cofficients
	  cx0 = 
	    (2.0 * getG(g_Row, g_Col, g_Sxx, level) 
	     + getG(g_Row, g_Col, g_Syy, level)) * dhdx 
	     + getG(g_Row, g_Col, g_Sxy, level) * dhdy;

	  cx1 = 
	    - (1.0 + alpha) * dhdx
	    + 2.0 * grad(g_Row, g_Col, g_Sxx    , 1, 0, level)   // d(Sxx)/dx
	    +        grad(g_Row, g_Col, g_Syy    , 1, 0, level)   // d(Syy)/dx
	    +        grad(g_Row, g_Col, g_Sxy    , 0, 1, level);  // d(Sxy)/dy

	  // Compute y-cofficients
	  cy0 = 
	    (getG(g_Row, g_Col, g_Sxx, level) 
	     + 2.0 * getG(g_Row, g_Col, g_Syy, level)) * dhdy 
 	     + getG(g_Row, g_Col, g_Sxy, level) * dhdx;

	  cy1 = 
	    - (1.0 + alpha) * dhdy
	    +        grad(g_Row, g_Col, g_Sxx    , 0, 1, level)   // d(Sxx)/dy
	    + 2.0 *  grad(g_Row, g_Col, g_Syy    , 0, 1, level)   // d(Syy)/dy
	    +        grad(g_Row, g_Col, g_Sxy    , 1, 0, level);  // d(Sxy)/dx
	  

	  if (H > 5.0) {
	  taud2 = 
	      getG(g_Row, g_Col, g_Sxx, level) * getG(g_Row, g_Col, g_Sxx, level)
	    + getG(g_Row, g_Col, g_Sxx, level) * getG(g_Row, g_Col, g_Syy, level)
	    + getG(g_Row, g_Col, g_Syy, level) * getG(g_Row, g_Col, g_Syy, level)
	    + getG(g_Row, g_Col, g_Sxy, level) * getG(g_Row, g_Col, g_Sxy, level);
	  }
	  
	  k0 = cx0*cx0 + cy0*cy0 + taud2;
	  k1 = 2.0 * (cx0*cx1 + cy0*cy1);
	  k2 = cx1*cx1 + cy1*cy1;
	  
	  if (UpdateWanted) {
	    Taue_suggested = (k0 + 0.5 * k1 * H + (k2 * H*H)/3.0);             // Used to compute residual
	    setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( Taue_suggested - getG(g_Row, g_Col, g_Current, level)), level);
	    
	  } else {
	    
	    Taue_suggested = (k0 + k1 * H/2.0 + ( k2 * H*H )/3.0);
	    Taue_error     = ( Taue_suggested - getG(g_Row, g_Col, g_Current, level) );
	    Taue_new       = getG(g_Row, g_Col, g_Current, level) + constMem.Relaxs*Taue_error;

	    setG(g_Row, g_Col, g_Current  , Taue_new      , level);
	    setG(g_Row, g_Col, g_Suggested, Taue_suggested, level); // Used to compute residual
	    setG(g_Row, g_Col, g_Error    , Taue_error    , level);

	  };
	 };
  #ifdef __CPU__
        }
    }
  #endif
};


  /**
   * Computes the Norm of a matrix and it's residual. Norm is defined as,
   * \verbatim embed:rst
   * .. math::
   *    \frac{fa - f(x)}{f(x)} << 1
   * \endverbatim
   *
   */
#ifdef __GPU__               
__global__
#endif
  void ComputeL2Norm( const ValueType *M, const ValueType *ResM, ValueType *Norm, int xShift, int yShift, int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  if (  g_Row == 1 && g_Col == 1 ) {
  #else
    int g_Col = PADDING;
    int g_Row = PADDING;
  #endif
    (*Norm) = 0.0;
    ValueType SumDiff = 0.0; /* Sum of difference (residual - Value)^2 */
    ValueType SumTot = 0.0; /* Sum of the actual value squared */

  for (int i = 1+xShift; i < constMem.XNum[level]-1; i++) {
    for (int j = 1+yShift; j < constMem.YNum[level]-1; j++) {
      SumDiff += (getG(j,i,ResM,level) - getG(j,i,M,level)) * (getG(j,i,ResM,level) - getG(j,i,M,level));
      SumTot += getG(j,i,ResM,level)*getG(j,i,ResM,level);
      //      SumTot += getG(j,i,ResM,level)*getG(j,i,ResM,level);
    };
  };
  (*Norm) = (SumDiff)/(SumTot+constMem.Gamma);
    //(*Norm) = sqrt(SumTot);
   #ifdef __GPU__
  }
  #endif
};

  /**
   * Summation over all rows in the current matrix M.
   *
   * @param g_M Matrix of current values (i.e. Vx, Vy or Taue)
   * @param g_MNew Matrix of suggested new values from iteration
   * @param g_Residual Vector to hold sum of each row
   * @param xShift Shift start col by a integer
   * @param yShift Shift start row by a integer
   * @param dimensions The dimension of the array to sum (i.e. Vector or 2D Matrix)
   * @param ComputeDiff Compute the Normal Sum of the difference between g_M and g_MNew
   * @param level Current multigrid level number
   */
#ifdef __GPU__
__global__ 
#endif
  void SumRows(const ValueType *g_M, const ValueType *g_MNew, ValueType *g_Residual, 
	       int xShift, int yShift, unsigned int dimensions, const int ComputeDiff, int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+xShift;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING+yShift; 

  if (g_Row < dimensions && g_Col == PADDING+xShift) {
  #else
    for (unsigned int g_Row = PADDING+yShift; g_Row < dimensions; g_Row++) { // Changed to PADDING
      int g_Col = PADDING+xShift;
  #endif
      ValueType Sum = 0.0; /* Sum of the actual value squared */
      int SumBound = 0;

      if (ComputeDiff == 2) {
	SumBound = constMem.YNum[level]-1;
      } else {
	SumBound = constMem.XNum[level]-1;
      };

      // First thread of each row sums over all columns
      for (int Col = g_Col; Col < SumBound; Col++) {
	if (ComputeDiff == 1) {
	  Sum += (getG(g_Row, Col, g_MNew, level) - getG(g_Row, Col, g_M, level)) 
	       * (getG(g_Row, Col, g_MNew, level) - getG(g_Row, Col, g_M, level));
	} else if (ComputeDiff == 2) {
	  Sum += getG(g_Row, Col, g_MNew, level); // Used when summing the vector of sums
	} else if (ComputeDiff == 3) {
	  Sum += getG(g_Row, Col, g_M, level) * getG(g_Row, Col, g_M, level);
	} else {
	  Sum += getG(g_Row, Col, g_MNew, level) * getG(g_Row, Col, g_MNew, level);
	};
      };


      //      printf("%f \n", Sum);
      // Save partial residuals vector
      g_Residual[g_Row] = Sum;
    };    
};

#ifdef __GPU__
__global__ 
#endif
  void SumRowsBlockwise(const ValueType *g_M, const ValueType *g_MNew, ValueType *g_Residual, 
			int xShift, int yShift, unsigned int dimensions, const int ComputeDiff, int level, int color) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+xShift;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING+yShift; 

  if (g_Row < dimensions && g_Col == PADDING+xShift) {
  #else
    for (unsigned int g_Row = PADDING+yShift; g_Row < dimensions; g_Row++) { // Changed to PADDING
      int g_Col = PADDING+xShift;
  #endif
      ValueType Sum = 0.0; /* Sum of the actual value squared */
      int SumBound = 0;

      if (ComputeDiff == 2) {
	SumBound = constMem.YNum[level]-1;
      } else {
	SumBound = constMem.XNum[level]-1;
      };

      // First thread of each row sums over all columns
      // Only each second cell is computed corisponding to color
      for (int Col = g_Col+color; Col < SumBound; Col += 2) {
	if (ComputeDiff == 1) {
	  Sum += (getG(g_Row, Col, g_MNew, level) - getG(g_Row, Col, g_M, level)) 
	       * (getG(g_Row, Col, g_MNew, level) - getG(g_Row, Col, g_M, level));
	} else if (ComputeDiff == 2) {
	  Sum += getG(g_Row, Col, g_MNew, level); // Used when summing the vector of sums
	} else if (ComputeDiff == 3) {
	  Sum += getG(g_Row, Col, g_M, level) * getG(g_Row, Col, g_M, level);
	} else {
	  Sum += getG(g_Row, Col, g_MNew, level) * getG(g_Row, Col, g_MNew, level);
	};
      };


      //      printf("%f \n", Sum);
      // Save partial residuals vector
      g_Residual[g_Row] = Sum;
    };    
};
  /**
   * Find the max value in a given row in the current matrix M.
   *
   * @param g_M Matrix of current values (i.e. Vx, Vy or Taue)
   * @param g_MaxInRow the max value from each row
   * @param level Current multigrid level number
   */
#ifdef __GPU__
__global__ 
#endif
  void MaxRows(const ValueType *g_M, ValueType *g_MaxInRow, int level) {
  #ifdef __GPU__
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (unsigned int g_Row = PADDING; g_Row < constMem.YNum[level]-1; g_Row++) { // Changed to PADDING
    for (unsigned int g_Col = PADDING; g_Col < constMem.XNum[level]-1; g_Col++) {
//      int g_Col = PADDING;
  #endif
      ValueType MaxValue = 1.0e-16; /* The max value in the row */

      // Each thread looks into all colums in one row 
      // Note: Data is stored in row-major
      for (int Col = g_Col; Col < constMem.XNum[level]-1; Col++) {
	if (getG(g_Row, Col, g_M, level) > MaxValue) {
	  MaxValue = getG(g_Row, Col, g_M, level);
	};
      };

      // Save partial residuals vector
      g_MaxInRow[g_Row] = MaxValue;

   #ifdef __CPU__
} 
#endif
  };
};


/**
 * Computes stresses based on the current values of strain rate and effective stress
 *
 * @param g_Sxx Global pointer to Sxx stress matrix
 * @param g_Syy Global pointer to Syy stress matrix
 * @param g_Sxy Global pointer to Sxy stress matrix
 * @param g_Exx Global pointer to Exx strain rates
 * @param g_Eyy Global pointer to Eyy strain rates
 * @param g_Exy Global pointer to Exy strain rates (This should be the mean of the four corner nodes)
 * @param g_Taue Global pointer to effective stress matrix
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
*/
#ifdef __GPU__               
__global__
#endif
void g_update_stress(ValueType *g_Sxx, ValueType *g_Syy, ValueType *g_Sxy, 
		     const ValueType *g_Exx, const ValueType *g_Eyy, const ValueType *g_Exy, 
		     const ValueType *g_Taue, const ValueType *g_IceTopo, const ValueType *g_BedTopo, 
		     const int level, const int color)   
{

  int g_Col, g_Row;
  ValueType t_Taue, t_Sxx, t_Syy, t_Sxy, H;

  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else

    for (g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

	H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); 

	if (H > 5.0) {
	  t_Taue = getG(g_Row, g_Col, g_Taue, level);
	  t_Sxx  = getG(g_Row, g_Col, g_Exx, level)/(constMem.A*t_Taue + constMem.Gamma);
	  t_Syy  = getG(g_Row, g_Col, g_Eyy, level)/(constMem.A*t_Taue + constMem.Gamma);
	  t_Sxy  = getG(g_Row, g_Col, g_Exy, level)/(constMem.A*t_Taue + constMem.Gamma);

	  setG(g_Row, g_Col, g_Sxx, t_Sxx, level);
	  setG(g_Row, g_Col, g_Syy, t_Syy, level);
	  setG(g_Row, g_Col, g_Sxy, t_Sxy, level);

	} else {
	  setG(g_Row, g_Col, g_Sxx,  0.0, level);
	  setG(g_Row, g_Col, g_Syy,  0.0, level);
	  setG(g_Row, g_Col, g_Sxy,  0.0, level);
	};
      };

  #ifndef __GPU__
    }
  #endif
  
  };


/**
 * Computes the Exy strain rate in the corner nodes.
 * In this function only the cross differential term is computed.
 * The remanding terms are computed in the g_Compute_strain_rate function.
 *
 * @param g_Vx Global pointer to x-velocity matrix
 * @param g_Vy Global pointer to y-velocity matrix
 * @param g_Exy Global pointer to Exy strain rate in corner node
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
*/
#ifdef __GPU__               
__global__
#endif
  void g_Compute_Exy_cross_terms (const ValueType *g_Vx, const ValueType *g_Vy,
				  ValueType *g_Exyc,
				  const int level)    
  {
  #ifdef __GPU__
  
    // First compute node is (2,2) because of velocity boundary conditions
    // First row and column should be set using BC.
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  
    if (  g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
#else
      for (int g_Col=PADDING;g_Col<constMem.XNum[level];g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level];g_Row++) {
#endif
	  
	  ValueType term = 0.5 * (gradVelocity(g_Row-1, g_Col  , g_Vx, 0, 1, level)  // d(Vx)/dy
				 +gradVelocity(g_Row  , g_Col-1, g_Vy, 1, 0, level)  // d(Vy)/dx
				  );
	  
	  setG(g_Row, g_Col, g_Exyc, term, level);
	  
#ifndef __GPU__
      }
#endif
  };
  };


/**
 * Computes topography for benchmarks.
 *
*/
#ifdef __GPU__               
__global__
#endif
  void g_Compute_test_topography(ValueType *g_Ice, ValueType *g_Bed,
                                 ValueType *g_dbdx, ValueType *g_dbdy, // Nodes in cell center
                                 ValueType *g_dhdx, ValueType *g_dhdy,
				 ValueType *g_dbdx_w, ValueType *g_dbdy_w, // Nodes at velocity cells
				 ValueType *g_dhdx_w, ValueType *g_dhdy_w, // i.e. on cell walls
                                 int mode, int level) {
    
  
  const double pi = 3.14159265358979;

  double XStp = constMem.XDim/((double) constMem.XNum[level]-3);
  double YStp = constMem.YDim/((double) constMem.YNum[level]-3);
  ValueType dhdx, dhdy, dbdx, dbdy;
  ValueType dhdx_w, dhdy_w, dbdx_w, dbdy_w;
  ValueType Ice, Bed;

#ifdef __GPU__
    
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 
    
    if (  g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
#else
      for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
	for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
#endif
      
	// Center node ice and bed
	Ice = -((g_Col-0.5)*XStp) * tan(0.5*2.0*pi/360.0);
	Bed =  Ice - 1000.0 + 500.0 * sin(((g_Col-0.5)*XStp)*2.0*pi/(constMem.XDim)) * sin(((g_Row-0.5)*YStp)*2.0*pi/(constMem.YDim));
      
	// Compute Gradients
	// First node is located in (0.5, 0.5)
	dhdx = - tan(2.0*pi*0.5/360.0);
	// First node is located in (0.0, 0.5)
	dhdx_w = - tan(2.0*pi*0.5/360.0);
	
	// First node is located in (0.5, 0.5)
	dbdx = dhdx + 500 * 2.0*pi/(constMem.XDim) * sin((g_Row-0.5)*YStp*2.0*pi/(constMem.YDim)) 
	  * cos(XStp*(g_Col-0.5)*2.0*pi/(constMem.XDim));
	// First node is located in (0.0, 0.5)
	dbdx_w = dhdx_w + 500 * 2.0*pi/(constMem.XDim) * sin((g_Row-0.5)*YStp*2.0*pi/(constMem.YDim)) 
	  * cos(XStp*(g_Col)*2.0*pi/(constMem.XDim));
	
	
	// First node is located in (0.5, 0.5)
	dbdy = 500 * 2.0*pi/(constMem.YDim) * sin(XStp*(g_Col-0.5)*2.0*pi/(constMem.XDim)) 
	  * cos((g_Row-0.5)*YStp*2.0*pi/(constMem.YDim));
	// First node is located in (0.5, 0.0)
	dbdy_w = 500 * 2.0*pi/(constMem.YDim-0*YStp) * sin(XStp*(g_Col-0.5)*2.0*pi/(constMem.XDim)) 
	  * cos((g_Row)*YStp*2.0*pi/(constMem.YDim));
	
	// First node is located in (0.5, 0.5)      
	dhdy = 0.0;
	// First node is located in (0.5, 0.0)
	dhdy_w = 0.0;
	
	
	// Copy cell center nodes
	
	setG(g_Row, g_Col, g_dhdx,dhdx, level);
	setG(g_Row, g_Col, g_dhdy,dhdy, level);
	setG(g_Row, g_Col, g_dbdx,dbdx, level);
	setG(g_Row, g_Col, g_dbdy,dbdy, level);
	
	setG(g_Row, g_Col, g_Bed, Bed, level);
	setG(g_Row, g_Col, g_Ice, Ice, level);    

	// Copy cell velocity nodes
	
	setG(g_Row, g_Col, g_dhdx_w,dhdx_w, level);
	setG(g_Row, g_Col, g_dhdy_w,dhdy_w, level);
	setG(g_Row, g_Col, g_dbdx_w,dbdx_w, level);
	setG(g_Row, g_Col, g_dbdy_w,dbdy_w, level);
	 
    }
#ifndef __GPU__
    };
#endif
      
};


/**
 * Computes strain rates from current velocities
 *
 * @param g_Vx Global pointer to x-velocity matrix
 * @param g_Vy Global pointer to y-velocity matrix
 * @param g_Vxs Global pointer to x-surface velocity matrix
 * @param g_Vys Global pointer to y-surface velocity matrix
 * @param g_Vxb Global pointer to x-sliding velocity matrix
 * @param g_Vyb Global pointer to y-sliding velocity matrix
 * @param g_Exx Global pointer to save Exx strain rates in
 * @param g_Eyy Global pointer to save Eyy strain rates in
 * @param g_Exy Global pointer to save Exy strain rates in
 * @param g_Ezz Global pointer to save Ezz strain rates in
 * @param level Current multigrid level
 * @param color Current color (Red/Black)
*/
#ifdef __GPU__               
__global__
#endif
void  g_Compute_strain_rate(ValueType *g_Vx,  ValueType *g_Vy,
			    ValueType *g_Vxs, ValueType *g_Vys,
			    ValueType *g_Vxb, ValueType *g_Vyb,
			    ValueType *g_Exx, ValueType *g_Eyy, 
			    ValueType *g_Exy, ValueType *g_Ezz,
			    ValueType *g_IceTopo, ValueType* g_BedTopo,
			    const ValueType *g_dhdx, const ValueType *g_dhdy,
			    const ValueType *g_dbdx, const ValueType *g_dbdy,
			    int level, int color)
{
  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif
	  

	ValueType H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); 
	
	if (H > 5.0) {
	  ValueType Vx  = (getG(g_Row, g_Col, g_Vx, level)  + getG(g_Row, g_Col+1, g_Vx, level))/2.0; // Mean Ice velocity in center node
	  ValueType Vxs = (getG(g_Row, g_Col, g_Vxs, level) + getG(g_Row, g_Col+1, g_Vxs, level))/2.0; // Ice surface velocity in center node
	  ValueType Vxb = (getG(g_Row, g_Col, g_Vxb, level) + getG(g_Row, g_Col+1, g_Vxb, level))/2.0; // Ice sliding velocity in center node
	  
	  ValueType Vy  = (getG(g_Row, g_Col, g_Vy, level)  + getG(g_Row+1, g_Col, g_Vy, level))/2.0; // Mean Ice velocity in center node
	  ValueType Vys = (getG(g_Row, g_Col, g_Vys, level) + getG(g_Row+1, g_Col, g_Vys, level))/2.0; // Ice surface velocity in center node
	  ValueType Vyb = (getG(g_Row, g_Col, g_Vyb, level) + getG(g_Row+1, g_Col, g_Vyb, level))/2.0; // Ice sliding velocity in center node
	  
	  ValueType dhdx = getG(g_Row, g_Col, g_dhdx, level);
	  ValueType dhdy = getG(g_Row, g_Col, g_dhdy, level);
	  ValueType dbdx = getG(g_Row, g_Col, g_dbdx, level);
	  ValueType dbdy = getG(g_Row, g_Col, g_dbdy, level);

	  ValueType t_Exx = gradVelocity( g_Row, g_Col, g_Vx, 1, 0, level)  // d(Vx)/dx
	    - (
	         dhdx * ( Vxs - Vx  )
	       + dbdx * ( Vx  - Vxb )
	       )/(H);
	  

	  ValueType t_Eyy = gradVelocity( g_Row, g_Col, g_Vy, 0, 1, level) // d(Vy)/dy 
	    - (
  	         dhdy * ( Vys - Vy  )
	       + dbdy * ( Vy  - Vyb )
	       )/(H);
	  
	  ValueType t_Exy = 
	    getG(g_Row, g_Col, g_Exy, level) - (
						  dhdy * ( Vxs - Vx  )
						+ dhdx * ( Vys - Vy  )
						+ dbdy * ( Vx  - Vxb )
						+ dbdx * ( Vy  - Vyb )
						)/(2.0*H);


	  if ((
	      (g_Col == 1 && g_Row == 1) ||
	      (g_Col == 1 && g_Row == 2) ||
	      (g_Col == 1 && g_Row == 3) ||
	      (g_Col == 1 && g_Row == 4) /*||
	      (g_Col == 5 && g_Row == 64)*/
	      ) && level == 0  && 0)
	    printf("g_Row %i g_Col %i - d(vx)/dx %e vx %e vx(0,1) %e dhdx %e dbdx %e dhdy %e dbdy %e exx %e Eyy %e Exy %e \n",g_Row, g_Col,gradVelocity( g_Row, g_Col, g_Vx, 1, 0, level),getG(g_Row, g_Col, g_Vx, level), getG(g_Row, g_Col+1, g_Vx, level),dhdx,dbdx,dhdy,dbdy,t_Exx,t_Eyy,t_Exy);

	  ValueType t_Ezz = -1.0 * (t_Exx + t_Eyy);
	  /*
	  #ifdef __CPU__
	    t_Exx *= -1;
	    t_Eyy *= -1;
	    t_Exy *= -1;
	    t_Ezz *= -1;
	  #endif
	  */
	  // Transfer back to global memory
	  setG(g_Row, g_Col, g_Exx, t_Exx, level);
	  setG(g_Row, g_Col, g_Eyy, t_Eyy, level);
	  setG(g_Row, g_Col, g_Exy, 1*t_Exy, level);
	  setG(g_Row, g_Col, g_Ezz, t_Ezz, level);
	} else {
	  // Transfer back to global memory
	  setG(g_Row, g_Col, g_Exx, 0.0, level);
	  setG(g_Row, g_Col, g_Eyy, 0.0, level);
	  setG(g_Row, g_Col, g_Exy, 0.0, level);
	  setG(g_Row, g_Col, g_Ezz, 0.0, level);
	};
      };
      
    #ifndef __GPU__
    };
    #endif
};


/**
 * Interpolates Exy strain rates from the corner nodes to
 * center nodes
 *
 * @param g_Exyc Global pointer to corner nodes
 * @param g_Exy Global pointer to center nodes to hold mean
 * @param level Current multigrid level
*/
#ifdef __GPU__
__global__
#endif
void g_Interp_Exy(const ValueType *g_Exyc, ValueType *g_Exy, int level)
{

  #ifdef __GPU__
  
  // The first Exy corner node is (2,2)
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

	ValueType t_Exym = ( getG(g_Row  , g_Col  , g_Exyc, level)
			   + getG(g_Row+1, g_Col  , g_Exyc, level)
			   + getG(g_Row  , g_Col+1, g_Exyc, level)
			   + getG(g_Row+1, g_Col+1, g_Exyc, level) // NOTE: This is the same color as the current node!
			   )/4.0;                                 // but a different memory space
    
    setG(g_Row, g_Col, g_Exy, t_Exym, level);

    #ifndef __GPU__
      }; // End of for-loops
   #endif
  };
};


/**
 * Interpolates values from the center of cells to the base nodes at the corner
 *
 * @param g_In Matrix to interpolate from
 * @param g_Out Matrix to interpolate too
 * @param level The current multigrid level
 */
#ifdef __GPU__
  __global__
#endif
    void g_CenterToCornerInterp(ValueType* g_In, ValueType* g_Out, const int level) {
    
#ifdef __GPU__
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
    
    if (  g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
#else
      for (int g_Col=PADDING;g_Col<constMem.XNum[level];g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level];g_Row++) {
#endif

	  ValueType Interp = (   getG(g_Row  , g_Col  , g_In, level)
			       + getG(g_Row-1, g_Col  , g_In, level)
			       + getG(g_Row  , g_Col-1, g_In, level)
			       + getG(g_Row-1, g_Col-1, g_In, level)
			       )/4.0;

	  setG(g_Row, g_Col, g_Out, Interp, level);

    #ifndef __GPU__
      }; // End of for-loops
   #endif    
  };
};
  

/**
 * Sets given boundary condition for velocities and stresses
 *
 * The boundaires are given in a North, South, East, West fasion like the following
 *   
 *         N
 *         |
 *    W <-----> E
 *         |
 *         S
 *
 * The grid starts in (0,0) that equals the N/W corner.
 *    
 *
 * @param g_Sxx Global pointer to device memory
 * @param g_Syy Global pointer to device memory
 * @param g_Sxy Global pointer to device memory
 * @param SetValues The matrix to set bc for
*/
#ifdef __GPU__
__global__
#endif
 void setBoundaries( ValueType* g_Vx, //!< Global pointer to x-velocity in device memory
		     ValueType* g_Vy, //!< Global pointer to y-velocity in device memory
		     ValueType* g_Vxs,
		     ValueType* g_Vys,
		     ValueType* g_Sxx, 
		     ValueType* g_Syy, 
		     ValueType* g_Sxy,
		     ValueType* g_Sxxm, 
		     ValueType* g_Syym, 
		     ValueType* g_Sxym,
		     ValueType* g_Exy,
		     ValueType* g_Exyc,
		     const int SetValues,
		     int level) {

  double interp = 0.0;

  #ifdef __GPU__
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y; 

    if ( g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
  #else
    for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
  #endif      


	
	if (constMem.BC[0] == 7) {
	    return;
	  }
	

  //////////////////////////////////
  //         WEST BOUNDARY        //
  //////////////////////////////////
  if (g_Col == 1) {
    
    if (constMem.BC[0] == 0) {
      // Free slip
      setG(g_Row,1,g_Vx,0,level); // Vx
      setG(g_Row,0,g_Vy,getG(g_Row,1,g_Vy,level),level); // Vy

    } else if (constMem.BC[0] == 1) {
      // No Slip
      setG(g_Row,1,g_Vx,0,level); // Vx
      setG(g_Row,0,g_Vy,-getG(g_Row,1,g_Vy,level),level); // Vy

    } else if (constMem.BC[0] == 2) {
      // Test case
      setG(g_Row,1,g_Vx,5,level);
      setG(g_Row,0,g_Vy,5,level);

    } else {

      // Periodic boundaries
      if (SetValues == 0) {

	setG(g_Row,1,g_Vx,getG(g_Row,constMem.XNum[level]-2,g_Vx,level),level); // Vx
	setG(g_Row,0,g_Vy,getG(g_Row,constMem.XNum[level]-2,g_Vy,level),level); // Vy
	setG(g_Row,1,g_Vxs,getG(g_Row,constMem.XNum[level]-2,g_Vxs,level),level); // Vx
	setG(g_Row,0,g_Vys,getG(g_Row,constMem.XNum[level]-2,g_Vys,level),level); // Vy

	setG(g_Row,0,g_Sxx,getG(g_Row,constMem.XNum[level]-2,g_Sxx,level),level); // Sxx
	setG(g_Row,0,g_Syy,getG(g_Row,constMem.XNum[level]-2,g_Syy,level),level); // Syy
	setG(g_Row,0,g_Sxy,getG(g_Row,constMem.XNum[level]-2,g_Sxy,level),level); // Sxy

	setG(g_Row,0,g_Sxxm,getG(g_Row,constMem.XNum[level]-2,g_Sxxm,level),level); // Sxx
	setG(g_Row,0,g_Syym,getG(g_Row,constMem.XNum[level]-2,g_Syym,level),level); // Syy
	setG(g_Row,0,g_Sxym,getG(g_Row,constMem.XNum[level]-2,g_Sxym,level),level); // Sxy

	if (g_Row == 1 && g_Col == 1) {
	  interp = (getG(1,0,g_Sxx,level)+getG(0,1,g_Sxx,level)+getG(1,1,g_Sxx,level))/3.0;
	  setG(0,0,g_Sxx,interp,level);
	  
	  interp = (getG(1,0,g_Syy,level)+getG(0,1,g_Syy,level)+getG(1,1,g_Syy,level))/3.0;
	  setG(0,0,g_Syy,interp,level);
	  
	  interp = (getG(1,0,g_Sxy,level)+getG(0,1,g_Sxy,level)+getG(1,1,g_Sxy,level))/3.0;
	  setG(0,0,g_Sxy,interp,level);
	};
      };

      if (SetValues == 1) {
	setG(g_Row,1,g_Exyc ,getG(g_Row,constMem.XNum[level]-2,g_Exyc,level),level); // Syy
      };

      if (SetValues == 2) {
	setG(g_Row,1,g_Exy ,getG(g_Row,constMem.XNum[level]-2,g_Exy,level),level); // Syy

	//	setG(g_Row,1,g_Exx ,getG(g_Row,constMem.XNum[level]-2,g_Exx,level),level); // Syy
	//	setG(g_Row,1,g_Eyy ,getG(g_Row,constMem.XNum[level]-2,g_Eyy,level),level); // Syy
      };
      /*
      setG(g_Row,0,g_Exx,getG(g_Row,constMem.XNum[level]-2,g_Exx,level),level); // Exx
      setG(g_Row,0,g_Eyy,getG(g_Row,constMem.XNum[level]-2,g_Eyy,level),level); // Eyy
      setG(g_Row,0,g_Exy,getG(g_Row,constMem.XNum[level]-2,g_Exy,level),level); // Exy
      */
    };
    
  } else {
    //////////////////////////////////
    //          EAST BOUNDARY       //
    //////////////////////////////////
    if (g_Col == constMem.XNum[level]-2) {

    if (constMem.BC[1] == 0) {
      // Free slip
      setG(g_Row,g_Col+1,g_Vx,0,level); // Vx
      setG(g_Row,g_Col+1,g_Vy,getG(g_Row,g_Col,g_Vy,level),level); // Vy

    } else if (constMem.BC[1] == 1) {
      // No Slip
      setG(g_Row,g_Col+1,g_Vx,0,level); // Vx
      setG(g_Row,g_Col+1,g_Vy,-getG(g_Row,g_Col,g_Vy,level),level); // Vy

    } else if (constMem.BC[1] == 2) {
	setG(g_Row,g_Col+1,g_Vx,5,level);
	setG(g_Row,g_Col+1,g_Vy,5,level);

    } else {

      // Periodic boundaries
      if (SetValues == 0) {

      setG( g_Row, constMem.XNum[level]-1, g_Vx, getG( g_Row, 2, g_Vx, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Vy, getG( g_Row, 1, g_Vy, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Vxs, getG( g_Row, 2, g_Vxs, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Vys, getG( g_Row, 1, g_Vys, level), level);

      setG( g_Row, constMem.XNum[level]-1, g_Sxx, getG( g_Row, 1, g_Sxx, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Syy, getG( g_Row, 1, g_Syy, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Sxy, getG( g_Row, 1, g_Sxy, level), level);

      setG( g_Row, constMem.XNum[level]-1, g_Sxxm, getG( g_Row, 1, g_Sxxm, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Syym, getG( g_Row, 1, g_Syym, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Sxym, getG( g_Row, 1, g_Sxym, level), level);


      if (g_Row == constMem.YNum[level]-2 && g_Col == constMem.XNum[level]-2) {
	interp = (getG(constMem.YNum[level]-2,constMem.XNum[level]-1,g_Sxx,level)
		  +getG(constMem.YNum[level]-1,constMem.XNum[level]-2,g_Sxx,level)
		  +getG(constMem.YNum[level]-2,constMem.XNum[level]-2,g_Sxx,level))/3.0;
	setG(constMem.YNum[level]-1,constMem.XNum[level]-1,g_Sxx,interp,level);
	
	interp = (getG(constMem.YNum[level]-2,constMem.XNum[level]-1,g_Syy,level)
		  +getG(constMem.YNum[level]-1,constMem.XNum[level]-2,g_Syy,level)
		  +getG(constMem.YNum[level]-2,constMem.XNum[level]-2,g_Syy,level)
		  )/3.0;
	setG(constMem.YNum[level]-1,constMem.XNum[level]-1,g_Syy,interp,level);
	
	interp = (getG(constMem.YNum[level]-2,constMem.XNum[level]-1,g_Sxy,level)
		  +getG(constMem.YNum[level]-1,constMem.XNum[level]-2,g_Sxy,level)
		  +getG(constMem.YNum[level]-2,constMem.XNum[level]-2,g_Sxy,level)
		  )/3.0;
	setG(constMem.YNum[level]-1,constMem.XNum[level]-1,g_Sxy,interp,level);
      };
      }
      if (SetValues == 1) {
	setG( g_Row, constMem.XNum[level]-1, g_Exyc, getG( g_Row, 2, g_Exyc, level), level);
      };

      if (SetValues == 2) {
	setG( g_Row, constMem.XNum[level]-1, g_Exy, getG( g_Row, 2, g_Exy, level), level);

	//	setG( g_Row, constMem.XNum[level]-1, g_Exx, getG( g_Row, 0, g_Exx, level), level);
	//	setG( g_Row, constMem.XNum[level]-1, g_Eyy, getG( g_Row, 0, g_Eyy, level), level);
      };
      /*
      setG( g_Row, constMem.XNum[level]-1, g_Exx, getG( g_Row, 1, g_Exx, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Eyy, getG( g_Row, 1, g_Eyy, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Exy, getG( g_Row, 1, g_Exy, level), level);
      */
      };
    };
  };

  ////////////////////////////////////
  //          NORTH BOUNDARY        //
  // Bottom of the figure in ncview //
  ////////////////////////////////////
  if (g_Row == 1) {
    if (constMem.BC[2] == 0) {
      // Free slip
      setG(0,g_Col,g_Vx,getG(1,g_Col,g_Vx,level),level); // Vx
      setG(1,g_Col,g_Vy,0,level); // Vy

    } else if (constMem.BC[2] == 1) {
      // No Slip
      setG(0,g_Col,g_Vx,-getG(g_Row,g_Col,g_Vx,level),level);
      setG(1,g_Col,g_Vy,0,level);

    } else if (constMem.BC[2] == 2) {
	setG(0,g_Col,g_Vx,5,level);
	setG(1,g_Col,g_Vy,5,level);

    } else {

      // Periodic boundaries
      if (SetValues == 0) {

      setG( 0, g_Col, g_Vx, getG( constMem.YNum[level]-2, g_Col, g_Vx, level), level);
      setG( 1, g_Col, g_Vy, getG( constMem.YNum[level]-2, g_Col, g_Vy, level), level);
      setG( 0, g_Col, g_Vxs, getG( constMem.YNum[level]-2, g_Col, g_Vxs, level), level);
      setG( 1, g_Col, g_Vys, getG( constMem.YNum[level]-2, g_Col, g_Vys, level), level);

      setG( 0, g_Col, g_Sxx, getG( constMem.YNum[level]-2, g_Col, g_Sxx, level), level);
      setG( 0, g_Col, g_Syy, getG( constMem.YNum[level]-2, g_Col, g_Syy, level), level);
      setG( 0, g_Col, g_Sxy, getG( constMem.YNum[level]-2, g_Col, g_Sxy, level), level);

      setG( 0, g_Col, g_Sxxm, getG( constMem.YNum[level]-2, g_Col, g_Sxxm, level), level);
      setG( 0, g_Col, g_Syym, getG( constMem.YNum[level]-2, g_Col, g_Syym, level), level);
      setG( 0, g_Col, g_Sxym, getG( constMem.YNum[level]-2, g_Col, g_Sxym, level), level);

      if (g_Row == 1 && g_Col == constMem.XNum[level]-2) {
	interp = (getG(1,constMem.XNum[level]-1,g_Sxx,level)
		  +getG(0,constMem.XNum[level]-2,g_Sxx,level)
		  +getG(1,constMem.XNum[level]-2,g_Sxx,level)
		  )/3.0;
	setG(0,constMem.XNum[level]-1,g_Sxx,interp,level);
	
	interp = (getG(1,constMem.XNum[level]-1,g_Syy,level)
		  +getG(0,constMem.XNum[level]-2,g_Syy,level)
		  +getG(1,constMem.XNum[level]-2,g_Syy,level)
		  )/3.0;
	setG(0,constMem.XNum[level]-1,g_Syy,interp,level);
	
	interp = (getG(1,constMem.XNum[level]-1,g_Sxy,level)
		  +getG(0,constMem.XNum[level]-2,g_Sxy,level)
		  +getG(1,constMem.XNum[level]-2,g_Sxy,level)
		  )/3.0;
	setG(0,constMem.XNum[level]-1,g_Sxy,interp,level);
      
      };
      }

      if (SetValues == 1) {
	setG( 1, g_Col, g_Exyc, getG( constMem.YNum[level]-2, g_Col, g_Exyc, level), level);
      }

      if (SetValues == 2) {
	setG( 1, g_Col, g_Exy, getG( constMem.YNum[level]-2, g_Col, g_Exy, level), level);

	//	setG( 1, g_Col, g_Exx, getG( constMem.YNum[level]-2, g_Col, g_Exx, level), level);
	//	setG( 1, g_Col, g_Eyy, getG( constMem.YNum[level]-2, g_Col, g_Exy, level), level);
      }

      };
    
  } else {
    //////////////////////////////////
    //        SOUTH BOUNDARY        //
    // Top of the figure in ncview  //
    //////////////////////////////////
    if (g_Row == constMem.YNum[level]-2) {
      if (constMem.BC[3] == 0) {
	// Free slip
	setG(g_Row+1,g_Col,g_Vx,getG(g_Row,g_Col,g_Vx,level),level); // Vx
	setG(g_Row+1,g_Col,g_Vy,0,level); // Vy
	
      } else if (constMem.BC[3] == 1) {
	// No Slip
	setG(g_Row+1,g_Col,g_Vx,-getG(g_Row,g_Col,g_Vx,level),level);
	setG(g_Row+1,g_Col,g_Vy,0,level);
	
      } else if (constMem.BC[3] == 2) {
	setG(g_Row+1,g_Col,g_Vx,5,level);
	setG(g_Row+1,g_Col,g_Vy,5,level);

      } else {

	// Periodic boundaries
      if (SetValues == 0) {

	setG( constMem.YNum[level]-1, g_Col, g_Vx, getG( 1, g_Col, g_Vx, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Vy, getG( 2, g_Col, g_Vy, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Vxs, getG( 1, g_Col, g_Vxs, level), level);
       	setG( constMem.YNum[level]-1, g_Col, g_Vys, getG( 2, g_Col, g_Vys, level), level);

	setG( constMem.YNum[level]-1, g_Col, g_Sxx, getG( 1, g_Col, g_Sxx, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Syy, getG( 1, g_Col, g_Syy, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Sxy, getG( 1, g_Col, g_Sxy, level), level);

	setG( constMem.YNum[level]-1, g_Col, g_Sxxm, getG( 1, g_Col, g_Sxxm, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Syym, getG( 1, g_Col, g_Syym, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Sxym, getG( 1, g_Col, g_Sxym, level), level);

	if (g_Row == constMem.YNum[level]-2 && g_Col == 1) {
	  interp = (getG(constMem.YNum[level]-1,1,g_Sxx,level)
		    +getG(constMem.YNum[level]-2,0,g_Sxx,level)
		    +getG(constMem.YNum[level]-2,1,g_Sxx,level)
		    )/3.0;
	  setG(constMem.YNum[level]-1,0,g_Sxx,interp,level);
	  
	  interp = (getG(constMem.YNum[level]-1,1,g_Syy,level)
		    +getG(constMem.YNum[level]-2,0,g_Syy,level)
		    +getG(constMem.YNum[level]-2,1,g_Syy,level)
		    )/3.0;
	  setG(constMem.YNum[level]-1,0,g_Sxx,interp,level);
	  
	  interp = (getG(constMem.YNum[level]-1,1,g_Sxy,level)
		    +getG(constMem.YNum[level]-2,0,g_Sxy,level)
		    +getG(constMem.YNum[level]-2,1,g_Sxy,level)
		    )/3.0;
	  setG(constMem.YNum[level]-1,0,g_Sxx,interp,level);
	};
      }

      if (SetValues == 1) {
	setG( constMem.YNum[level]-1, g_Col, g_Exyc, getG( 2, g_Col, g_Exyc, level), level);
      };

      if (SetValues == 2) {
	setG( constMem.YNum[level]-2, g_Col, g_Exy, getG( 2, g_Col, g_Exy, level), level);

	//	setG( constMem.YNum[level]-1, g_Col, g_Exx, getG( 0, g_Col, g_Exx, level), level);
	//	setG( constMem.YNum[level]-1, g_Col, g_Eyy, getG( 0, g_Col, g_Eyy, level), level);
      };
      /*
	setG( constMem.YNum[level]-1, g_Col, g_Exx, getG( 1, g_Col, g_Exx, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Eyy, getG( 1, g_Col, g_Eyy, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Exy, getG( 1, g_Col, g_Exy, level), level);
      */
      };
    };
    

  }
      
      }
      
 #ifndef __GPU__
    };
 #endif

};

    /*-------------------------------------------------
           Hydrology
     --------------------------------------------------*/


#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_flux(
			       ValueType *g_Qxc, ValueType *g_Qyc, ValueType *g_qxs, ValueType *g_qys,
			       const ValueType *g_Psi, const ValueType *g_Sc, const ValueType *g_hw,
			       const ValueType *g_IceTopo, const ValueType *g_BedTopo,
			       const int level, const int color, const bool UpdateWanted
)   
{

  ValueType hwx = 0, hwy = 0, 
    qxs= 0, qys = 0,
    Qxc = 0, Qyc = 0,
    H = 0.0;
  int g_Col, g_Row;

  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

	   H = getG(g_Row,g_Col, g_IceTopo, level) - getG(g_Row,g_Col, g_BedTopo, level);

	   if ( 1.0 < H && 0.0 < getG(g_Row, g_Col, g_hw, level)  ) {
	     hwx = (getG(g_Row, g_Col, g_hw, level) + getG(g_Row, g_Col-1, g_hw, level))/2;
	     hwy = (getG(g_Row, g_Col, g_hw, level) + getG(g_Row-1, g_Col, g_hw, level))/2;
	     //printf("hwx: %f hwy: %f \n",hwx,hwy);
	     
	     // Compute sheet flux
	     qxs = - constMem.ks*pow(hwx,3.0)/(constMem.rho_w*constMem.g) *
	       ( gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level) );
	     
	     qys = - constMem.ks*pow(hwy,3.0)/(constMem.rho_w*constMem.g) *
	       ( gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level) );

	     // Compute channel flux
	     // Discharge [m^3/s]
	     Qxc = fabs( gradVelocity(g_Row, g_Col, g_Psi, 1, 0, level) - gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level)) + 1e-8;
	     Qxc = - constMem.kc * pow(getG(g_Row,g_Col, g_Sc, level),5/4) * pow(Qxc,-0.5) *
	       ( gradVelocity(g_Row, g_Col, g_Psi, 1, 0, level) - gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level) );

	     Qyc = fabs( gradVelocity(g_Row, g_Col, g_Psi, 0, 1, level) - gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level))+1e-8;
	     Qyc = - constMem.kc * pow(getG(g_Row,g_Col, g_Sc, level),5/4) * pow(Qyc,-0.5) *
	       ( gradVelocity(g_Row, g_Col, g_Psi, 0, 1, level) - gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level) );
	     
	     
	     /*
	     if (qxs > 0.0 || qys > 0.0)
	       printf("hwx: %e hwy: %e qxs %e qys %e Qxc %e Qyc %e dPsidx %e dPsidy %e \n",hwx,hwy,qxs,qys,Qxc,Qyc, 
		      gradVelocity(g_Row, g_Col, g_Psi, 1, 0, level) - gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level),
		      gradVelocity(g_Row, g_Col, g_Psi, 0, 1, level) - gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level));
	     */
	     setG(g_Row, g_Col, g_Qxc, Qxc, level);
	     setG(g_Row, g_Col, g_Qyc, Qyc, level);
	     setG(g_Row, g_Col, g_qxs, qxs, level);
	     setG(g_Row, g_Col, g_qys, qys, level);

	     
	   }	   
	   
    #ifndef __GPU__
      }; // End of for-loops
   #endif    
  };

};


#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_h(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, ValueType *g_old,
			    const ValueType *g_PwOld, const ValueType *g_Pw,
			    const ValueType *g_Qcx, const ValueType *g_qsx,
			    const ValueType *g_Qcy, const ValueType *g_qsy, 
			    const ValueType *g_Psi, const ValueType *g_mcx, const ValueType *g_mcy, 
			    const ValueType *g_R, const ValueType dt, const ValueType *g_Sc,
			    const ValueType *g_tb, const ValueType *g_vb,
			    const ValueType *g_IceTopo, const ValueType *g_BedTopo,
			    const int level, const int color, const bool UpdateWanted)   
{

  ValueType hw, dq, dQ, ms, mc, H, R;
  ValueType h_suggested, h_error, h_new;
  int g_Col, g_Row;

  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

	   H = getG(g_Row,g_Col, g_IceTopo, level) - getG(g_Row,g_Col, g_BedTopo, level);

	   if ( 10.0 < H) {
	     ms = (constMem.GeoFlux + getG(g_Row, g_Col, g_tb, level)*constMem.rho_i*constMem.g * getG(g_Row, g_Col, g_vb, level)*3.171e-8)/(constMem.rho_w*constMem.latentheat);
	     mc = 0.0;//getG(g_Row, g_Col, g_mcx, level)/2 + getG(g_Row, g_Col+1, g_mcx, level)/2 + getG(g_Row, g_Col, g_mcy, level)/2 + getG(g_Row+1, g_Col, g_mcy, level)/2;
	     dq = (getG(g_Row, g_Col+1, g_qsx, level) - getG(g_Row, g_Col, g_qsx, level))/constMem.Dx[level] + (getG(g_Row+1, g_Col, g_qsy, level) - getG(g_Row, g_Col, g_qsy, level))/constMem.Dy[level];
	     dQ = 0.0;//(getG(g_Row, g_Col, g_Qcx, level) - getG(g_Row, g_Col+1, g_Qcx, level)) + (getG(g_Row, g_Col, g_Qcy, level) - getG(g_Row+1, g_Col, g_Qcy, level));
	     

	  if (UpdateWanted) {
	    ValueType suggested = - ( dq + dQ )*dt + (mc + ms + 0.1 )*dt ;             // Used to compute residual
	    setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( suggested - getG(g_Row, g_Col, g_Current, level)), level);
	    
	  } else {


	    h_suggested = getG(g_Row, g_Col, g_old, level) 	      
	      - (dq + dQ)*dt 
	      - (10+1e-4)/(constMem.rho_w*constMem.g) * (getG(g_Row,g_Col,g_Pw,level) - getG(g_Row,g_Col,g_PwOld,level))/dt	      
	      + ( ms + getG(g_Row, g_Col, g_R, level) )*dt ;             // Used to compute residual
	    h_error     = getG(g_Row, g_Col, g_Wanted, level) - ( h_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	    //h_error     = h_suggested - getG(g_Row,g_Col,g_Suggested,level);
	    h_new       = (1.0 - constMem.Relaxh)*getG(g_Row, g_Col, g_Current, level) + constMem.Relaxh*h_suggested;

	        
	    if (h_new < 0.0) {
	      h_new = 1e-3;
	      h_suggested = 0.0;
	      h_error = 0.0;
	    }
	    

	    if (g_Row == 130 && g_Col == 130 && 0)
	      printf("h_wanted %e h_new: %e h_sugg: %e h_err %e ms %e R %e mc %e dq %e dQ %e pw %e \n",getG(g_Row, g_Col, g_Wanted,level), h_new,h_suggested,h_error,ms,getG(g_Row, g_Col, g_R,level),mc,dq,dQ,getG(g_Row,g_Col,g_Pw,level));

	    setG(g_Row, g_Col, g_Current  , h_new      , level); // Set new current Tau_e based on updated coefficients
	    setG(g_Row, g_Col, g_Suggested, h_suggested, level); // Used to compute residual
	    setG(g_Row, g_Col, g_Error    , h_error    , level);	   
	  }
	  } else {
	    setG(g_Row, g_Col, g_Current  , 0.0      , level); // Set new current Tau_e based on updated coefficients
	    setG(g_Row, g_Col, g_Suggested, 0.0, level); // Used to compute residual
	    setG(g_Row, g_Col, g_Error    , 0.0    , level);
	  }

	   
    #ifndef __GPU__
      }; // End of for-loops
   #endif    
  };
};


#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_sx(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			     ValueType *g_M,
			     const ValueType *g_Psi, const ValueType *g_Pw, const ValueType *g_Sc,			     
			     const ValueType *g_Qcx, const ValueType *g_qsx, const ValueType *g_tn,
			     const ValueType dt, 
			    const ValueType *g_IceTopo, const ValueType *g_BedTopo,
			     const int level, const int color, const bool UpdateWanted)   
{

  ValueType M, N, H;
  ValueType s_suggested, s_error, s_new;
  int g_Col, g_Row;


  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

	   H = getG(g_Row,g_Col, g_IceTopo, level) - getG(g_Row,g_Col, g_BedTopo, level);

	   if ( 1.0 < H ) {
	   M = (fabs(getG(g_Row, g_Col, g_Qcx, level)*gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level))
		+ constMem.lambdac * fabs(getG(g_Row, g_Col, g_qsx, level)*gradVelocity(g_Row, g_Col-1, g_Psi, 1, 0, level)))
		/(constMem.rho_w * constMem.latentheat);
	   
	   N = getG(g_Row, g_Col, g_tn, level) - getG(g_Row, g_Col, g_Pw, level);
	   
	  if (UpdateWanted) {
	    ValueType suggested = (constMem.rho_w/constMem.rho_i * M - 2 * constMem.A/27 * getG(g_Row, g_Col, g_Sc, level) * pow(fabs(N),2) * N)*dt;
	    setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( suggested - getG(g_Row, g_Col, g_Current, level)), level);
	    
	  } else {
	    
	    // Tau_Old - Tau_New
	    s_suggested = (constMem.rho_w/constMem.rho_i * M - 2 * constMem.A/27 * getG(g_Row, g_Col, g_Sc, level) * pow(fabs(N),2) * N)*dt;             // Used to compute residual
	    s_error     = getG(g_Row, g_Col, g_Wanted, level) - ( s_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	    s_new       = getG(g_Row, g_Col, g_Current, level) - constMem.Relaxh*s_error;


	    //	  if (g_Row == 2 && g_Col == 1)
	    //	    printf("%e %e %e %e \n", s_new,s_error,s_suggested,M);

	    setG(g_Row, g_Col, g_Current  , s_new      , level); // Set new current Tau_e based on updated coefficients
	    setG(g_Row, g_Col, g_Suggested, s_suggested, level); // Used to compute residual
	    setG(g_Row, g_Col, g_Error    , s_error    , level);
	    setG(g_Row, g_Col, g_M, M, level);
	  };	   
	   };
    #ifndef __GPU__
      }; // End of for-loops
   #endif    
  };
};


#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_sy(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error,
			     ValueType * g_M,
			     const ValueType *g_Psi, const ValueType *g_Pw, const ValueType *g_Sc,			     
			     const ValueType *g_Qc, const ValueType *g_qs, const ValueType *g_tn,
			     const ValueType dt, 
			     const ValueType *g_IceTopo, const ValueType *g_BedTopo,
			     const int level, const int color, const bool UpdateWanted)   
{

  ValueType M, N, psi,H;
  ValueType s_suggested, s_error, s_new;
  int g_Col, g_Row;


  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif


	   M = (fabs(getG(g_Row, g_Col, g_Qc, level)*gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level))
		+ constMem.lambdac * fabs(getG(g_Row, g_Col, g_qs, level)*gradVelocity(g_Row-1, g_Col, g_Psi, 0, 1, level)))
		/(constMem.rho_w * constMem.latentheat);
	   
	   N = getG(g_Row, g_Col, g_tn, level) - getG(g_Row, g_Col, g_Pw, level);
	   
	  if (UpdateWanted) {
	    ValueType suggested = (constMem.rho_w/constMem.rho_i * M - 2 * constMem.A/27 * getG(g_Row, g_Col, g_Sc, level) * pow(fabs(N),2) * N)*dt;
	    setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( suggested - getG(g_Row, g_Col, g_Current, level)), level);
	    
	  } else {
	    
	    // Tau_Old - Tau_New
	    s_suggested = (constMem.rho_w/constMem.rho_i * M - 2 * constMem.A/27 * getG(g_Row, g_Col, g_Sc, level) * pow(fabs(N),2) * N)*dt;             // Used to compute residual
	    s_error     = getG(g_Row, g_Col, g_Wanted, level) - ( s_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	    s_new       = getG(g_Row, g_Col, g_Current, level) - constMem.Relaxh*s_error;


	    //	  if (g_Row == 2 && g_Col == 1)
	    // printf("%e %e %e %e %e %e %e %e %e %e %e \n", alpha,cx1,cy1,dhdx,dhdy,k0,k1,k2,Taue_new,Taue_error,Taue_suggested);

	    setG(g_Row, g_Col, g_Current  , s_new      , level); // Set new current Tau_e based on updated coefficients
	    setG(g_Row, g_Col, g_Suggested, s_suggested, level); // Used to compute residual
	    setG(g_Row, g_Col, g_Error    , s_error    , level);
	    setG(g_Row, g_Col, g_M, M, level);
        };	   
	   
    #ifndef __GPU__
      }; // End of for-loops
   #endif    
  };
};


#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_pressure(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, ValueType *g_old,
				   const ValueType *g_hw, const ValueType *g_tn,
				   const ValueType *g_tb, const ValueType *g_vb,
				   const ValueType *g_qsx, const ValueType *g_qsy,
				   const ValueType *g_R, const double dt,
				   const ValueType *g_dbdx, const ValueType *g_dbdy,
				   const ValueType *g_Ice, const ValueType *g_Bed,
				   const int level, const int color, const bool UpdateWanted)   
{

  ValueType pw_suggested, pw_error, pw_new, N,dq,m,Vo,Vc;
  int g_Col, g_Row;


  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif
	   
	   if ( 10.0 < (getG(g_Row,g_Col,g_Ice,level) - getG(g_Row,g_Col,g_Bed,level)) ) {	       

	   m = (constMem.GeoFlux + getG(g_Row, g_Col, g_tb, level)*constMem.rho_i*constMem.g * getG(g_Row, g_Col, g_vb, level)*3.171e-8)/(constMem.rho_w*constMem.latentheat);
	   N = getG(g_Row,g_Col,g_tn,level)*constMem.rho_i*constMem.g  - getG(g_Row,g_Col,g_Current,level);

	   // Limit effective pressure
	   if (N < 0.0 && 0)
	     N = getG(g_Row,g_Col,g_tn,level)*constMem.rho_i*constMem.g*0.95;

	   //	   dq = (getG(g_Row, g_Col+1, g_qsx, level) + getG(g_Row, g_Col, g_qsx, level))/2.0 + (getG(g_Row+1, g_Col, g_qsy, level) + getG(g_Row, g_Col, g_qsy, level))/2.0;
	   dq = gradVelocity(g_Row,g_Col,g_qsx,1,0,level) + gradVelocity(g_Row,g_Col,g_qsy,0,1,level);
	   Vo =  getG(g_Row,g_Col, g_vb, level)*3.171e-8*(constMem.hr - getG(g_Row,g_Col, g_hw, level))/constMem.lr;
	   Vc = (2*6.8e-24/27) * getG(g_Row,g_Col,g_hw,level) * pow(N,3);

	     if (UpdateWanted) {
	       ValueType suggested = pow(getG(g_Row, g_Col, g_hw, level)/constMem.hc, -1/constMem.gamma_h) * getG(g_Row, g_Col, g_tn, level);
	       setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( suggested - getG(g_Row, g_Col, g_Current, level)), level);
	       
	     } else {

	       // Poroelastic sheet

	       pw_suggested = getG(g_Row,g_Col,g_tn,level)*constMem.rho_i*constMem.g*getG(g_Row,g_Col,g_hw,level)/constMem.hr;
	       pw_new   = (1.0 - constMem.Relaxh)*getG(g_Row, g_Col, g_Current, level) + constMem.Relaxh*pw_suggested;

	       	            
	       // Cavities
	       /*
	       pw_suggested = 
		 - constMem.rho_w/constMem.rho_i * m 
		 - Vo + Vc - dq		 
		 //		 - (10+1e-4)/(constMem.rho_w*constMem.g) * (getG(g_Row,g_Col,g_Current,level) - getG(g_Row,g_Col,g_old,level))/dt
		 + m + getG(g_Row, g_Col, g_R, level) + getG(g_Row,g_Col,g_Current,level);
	       pw_new       = (1.0 - constMem.Relaxh)*getG(g_Row, g_Col, g_Current, level) + constMem.Relaxh*pw_suggested;
	       */

	       /*
	       pw_suggested = pw_suggested/(constMem.ks*pow(getG(g_Row,g_Col,g_hw,level),3)/(constMem.g*constMem.rho_w));
	       pw_suggested = constMem.Dx[level]/2*(pw_suggested
						    +grad(g_Row,g_Col,g_Current,1,0,level)
						    +grad(g_Row,g_Col,g_Current,0,1,level)
						    -2*getG(g_Row,g_Col,g_Current,level)/constMem.Dy[level]
						    );
	       */
	       /*
	       ValueType J = ((6*6.8e-24/27) * getG(g_Row,g_Col,g_hw,level) * pow(getG(g_Row,g_Col,g_Current,level),2)
	       -constMem.ks*pow(getG(g_Row,g_Col,g_hw,level),3)/(constMem.rho_w*constMem.g)*(-2/constMem.Dx[level]-2/constMem.Dy[level]));
	       ValueType Jdiff = (12*6.8e-24/27) * getG(g_Row,g_Col,g_hw,level) * getG(g_Row,g_Col,g_Current,level);
	       (pw_suggested - getG(g_Row,g_Col,g_Suggested,level))/(getG(g_Row,g_Col,g_Current,level) - getG(g_Row,g_Col,g_old,level));
	       */

	       // Newton Method
	       //	       pw_new       = getG(g_Row, g_Col, g_old, level) - pw_suggested*J/(J*J-pw_suggested*Jdiff);
	       //	       pw_new       = getG(g_Row, g_Col, g_old, level) - pw_suggested/J;
	       //	       pw_new       = getG(g_Row, g_Col, g_old, level) - pw_suggested/J;

	       pw_error     = getG(g_Row, g_Col, g_Wanted, level) - ( pw_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	       //pw_error = pw_suggested - getG(g_Row,g_Col,g_Suggested,level);
	       //	       pw_error = abs(pw_new - getG(g_Row,g_Col,g_Current,level));



	       if (pw_new < 0.0) {
		 pw_suggested = 0.0;
		 pw_new = 0.0;
	       }
	       

	       if (pw_new != 0.0 && 0)
		 printf("m %e N %e dq %e Vo %e Vc %e hw %e tn %e sugg %e error %e new %e \n",m,N,dq,Vo,Vc,getG(g_Row,g_Col,g_hw,level),getG(g_Row,g_Col,g_tn,level),pw_suggested,pw_error,pw_new);
	    
	       setG(g_Row, g_Col, g_Current  , pw_new      , level); // Set new current Tau_e based on updated coefficients
	       setG(g_Row, g_Col, g_Suggested, pw_suggested, level); // Used to compute residual
	       setG(g_Row, g_Col, g_Error    , pw_error    , level);
	       
	     };
	   };	   
	     
#ifndef __GPU__
      }; // End of for-loops
   #endif    
  };
};

#ifdef __GPU__               
__global__
#endif
  void g_hydrology_update_psi(ValueType *g_Psi, 
			      const ValueType *g_bed, 
			      const ValueType *g_Pw,
			      const int level)   
{

  int g_Col, g_Row;
  ValueType psi;

  #ifdef __GPU__
  
  g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	 if ((g_Row*g_Col % 2) == color) {
  #endif

	   psi = constMem.rho_w*constMem.g*getG(g_Row,g_Col,g_bed,level) 
	     + getG(g_Row,g_Col,g_Pw,level);
	   setG(g_Row, g_Col, g_Psi, 
	     psi, level); // Set new current Tau_e based on updated coefficients
	   //	   if (psi > 0.0)
	     //	     printf("psi %e\n",psi);
	   
    #ifndef __GPU__
      }; // End of for-loops
    }
   #endif    
  };
};

    
#endif






