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
ValueType getG( int Row, int Col, const ValueType *M, int level)
{
  // Get element of the array M that has M(Col,Row)
  return M[Row*(constMem.XNum[level])+Col];
};

#ifdef __GPU__               
__device__
#endif
void setG( int Row, int Col, ValueType *M, const ValueType Val, int level)
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
  /*
  ValueType interp1 = 0.0;
  ValueType interp2 = 0.0;


  // Gradient 1
  interp1 = ( (getG(Row+xDisplacement, Col+yDisplacement, M, level)) - (getG(Row, Col, M, level)) )/(((ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]));

  // Gradient 2
  interp2 = ( (getG(Row, Col, M, level)) - (getG(Row-xDisplacement, Col-yDisplacement, M, level)) )/(((ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]));

*/  
  interp = ( (getG(Row+xDisplacement, Col+yDisplacement, M, level)) - (getG(Row-xDisplacement, Col-yDisplacement, M, level)) )/(2.0 * ((ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]));
  

      //interp=(interp1+interp2)*0.5;
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
  interp = ( (getG(Row+xDisplacement, Col+yDisplacement, M, level)) - (getG(Row, Col, M, level)) )/( (ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]);

  //  interp2 = ( (getG(Row+xDisplacement, Col+yDisplacement, M, level)) - (getG(Row, Col, M, level)) )/( (ValueType)(xDisplacement)*constMem.Dx[level] + (ValueType)(yDisplacement)*constMem.Dy[level]);

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
    if ( g_Row==UB && UB!=0) { 
      // Halos above
      setG(g_Row,g_Col,g_M,0,level);
    };
    
    if ( g_Row==BB && BB!=0) { 
      // Halos below
      setG(g_Row,g_Col,g_M,0,level);
    };
    
    if (g_Col==LB && LB!=0) { 
      // Halos left
      setG(g_Row,g_Col,g_M,0,level);
    };
    
    if (g_Col==RB && RB!=0) { 
      // Halos right
      setG(g_Row,g_Col,g_M,0,level);
    };

    /*
    // LOAD CORNERS
    
    // Load upper left corners
    // Done by upper left thread
    if (g_Col == LB && g_Row == UB ) {
      setG(g_Row-1,g_Col-1,g_M,0,level);
    };
    
    // Load upper right corners
    // Done by upper right thread
    if (g_Col == RB && g_Row == UB ) {
      setG(g_Row-1,g_Col+1,g_M,0,level);
    };
    
    // Load lower left corners
    // Done by lower left thread
    if (g_Col == LB && g_Row == BB ) {
      setG(g_Row+1,g_Col-1,g_M,0,level);
    };
    
    // Load lower right corners
    // Done by lower right thread
    if (g_Col == RB && g_Row == BB ) {
      setG(g_Row+1,g_Col+1,g_M,0,level);
    };
    */
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

  int f_Row, f_Col, c_Col, c_Row, EvenCol;
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
    int  IsEvenRow, IsEvenCol, f_Col_odd, f_Col_even;
      
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
  #else
    for (int g_Col=0;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=0;g_Row<constMem.YNum[level];g_Row++) {
  #endif
  
  if ( g_Col < constMem.XNum[level] && g_Row < constMem.YNum[level]) {
    setG(g_Row, g_Col, g_M, 0.0, level);
  };

  #ifndef __GPU__
      };
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
    for (int g_Col=0;g_Col<XDim;g_Col++) {
      for (int g_Row=0;g_Row<YDim;g_Row++) {
  #endif
  
	//	cuPrintf("Set g_Row %i, g_Col %i to Value %f", g_Row, g_Col, Value);
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
  int OldWeight = 1.0;

  if ( Row_f < constMem.YNum[CoarseLevel-1]-1 && Col_f < constMem.XNum[CoarseLevel-1]-1 ) {
    
    int Row_c, Col_c; // Position in both arrays
    ValueType xf, yf, x1, y1, x2, y2, dxdx, dydy; // Physical position

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
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
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


#ifdef __GPU__               
__global__
#endif
  void cuGetGradients(ValueType* GradMat, ValueType *ValueMat, int x, int y, int color, int level) {

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    int g_Col = 0;
    int g_Row = 0;
    for (g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

	setG(g_Row, g_Col, GradMat,
	     grad(g_Row, g_Col, ValueMat, x, y, level), level);

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
  void g_update_vx(ValueType *g_Current, ValueType *g_Suggested, ValueType *g_Wanted, ValueType *g_Error, ValueType *g_Vxs,
		   const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
		   const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
		   const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
		   const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted)   
{

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	if ((g_Row*g_Col % 2) == color) {
  #endif

	if (g_Col > 1) {

	// Interpolate stresses to velocity node position
	ValueType Sxx = getG(g_Row, g_Col, g_Sxx, level);
	Sxx = getG(g_Row-1, g_Col, g_Sxx, level);
	ValueType Syy = getG(g_Row, g_Col, g_Syy, level);
	Syy = getG(g_Row-1, g_Col, g_Syy, level);
	ValueType Sxy = getG(g_Row, g_Col, g_Sxy, level);
	Sxy = getG(g_Row-1, g_Col, g_Sxy, level);
	
	ValueType cx0 = getG(g_Row, g_Col, g_IceTopo, level);
	cx0 = getG(g_Row-1, g_Col, g_IceTopo, level);
	cx0 = getG(g_Row, g_Col, g_IceTopom, level);
	cx0 = getG(g_Row, g_Col+1, g_IceTopom, level);
/*
	  (2.0 * Sxx + Syy)
	  * gradVelocity(g_Row-1, g_Col, g_IceTopo , 1, 0, level)       // d(h)/dx
	  + Sxy
	  * gradVelocity(g_Row  , g_Col, g_IceTopom, 0, 1, level);      // d(h)/dy
*/	
	ValueType cx1 = getG(g_Row, g_Col, g_IceTopo, level);
	cx1 = getG(g_Row-1, g_Col, g_IceTopo, level);
	cx1 = getG(g_Row, g_Col, g_Sxx, level);
	cx1 = getG(g_Row-1, g_Col, g_Sxx, level);
	cx1 = getG(g_Row, g_Col, g_Syy, level);
	cx1 = getG(g_Row-1, g_Col, g_Syy, level);
	cx1 = getG(g_Row, g_Col, g_Sxym, level);
	cx1 = getG(g_Row, g_Col+1, g_Sxym, level);
/*
	  - 1.0 * gradVelocity(g_Row-1, g_Col, g_IceTopo, 1, 0, level)  // d(h)/dx
	  + 2.0 * gradVelocity(g_Row-1, g_Col, g_Sxx    , 1, 0, level)  // d(Sxx)/dx
	  +       gradVelocity(g_Row-1, g_Col, g_Syy    , 1, 0, level)  // d(Syy)/dx
	  +       gradVelocity(g_Row  , g_Col, g_Sxym   , 0, 1, level); // d(Sxy)/dy
*/	
	// Compute y-cofficients in velocity position
	ValueType cy0 = getG(g_Row, g_Col, g_IceTopom, level);
		cy0 = getG(g_Row+1, g_Col, g_IceTopom, level);
		cy0 = getG(g_Row, g_Col, g_IceTopo, level);
		cy0 = getG(g_Row-1, g_Col, g_IceTopo, level);
/*
	  ( Sxx + 2.0 * Syy)
	  * gradVelocity(g_Row  , g_Col, g_IceTopom, 0, 1, level)       // d(h)/dy
	  + Sxy 
	  * gradVelocity(g_Row-1, g_Col, g_IceTopo , 1, 0, level);      // d(h)/dx
*/	
	ValueType cy1 = getG(g_Row, g_Col, g_IceTopom, level);
	cy1 = getG(g_Row, g_Col+1, g_IceTopom, level);
	cy1 = getG(g_Row, g_Col, g_Sxx, level);
	cy1 = getG(g_Row, g_Col+1, g_Sxxm, level);
	cy1 = getG(g_Row, g_Col, g_Syy, level);
	cy1 = getG(g_Row, g_Col+1, g_Syym, level);
	cy1 = getG(g_Row, g_Col, g_Sxy, level);
	cy1 = getG(g_Row-1, g_Col, g_Sxy, level);
/*
	  - 1.0 * gradVelocity(g_Row  , g_Col, g_IceTopom, 0, 1, level)  // d(h)/dy
	  +       gradVelocity(g_Row  , g_Col, g_Sxxm    , 0, 1, level)  // d(Sxx)/dy
	  + 2.0 * gradVelocity(g_Row  , g_Col, g_Syym    , 0, 1, level)  // d(Syy)/dy
	  +       gradVelocity(g_Row-1, g_Col, g_Sxy     , 1, 0, level); // d(Sxy)/dx
*/	
	ValueType taud2 =  Sxx*Sxx + Syy*Syy + Sxx*Syy + Sxy*Sxy;
	
	ValueType k0 = cx0 + cy0 + taud2;
	ValueType k1 = (cx0*cx1 + cy0*cy1);
	ValueType k2 = cx1 + cy1;
	
	ValueType H = getG(g_Row, g_Col, g_IceTopo, level);
	H = getG(g_Row, g_Col, g_BedTopo, level);
	H = getG(g_Row-1, g_Col, g_IceTopo, level);
	H = getG(g_Row-1, g_Col, g_BedTopo, level);
	
	// x-coefficients
	ValueType wx1 = (cx0 * k0);
	ValueType wx2 = (cx0*k1 + cx1*k0);
	ValueType wx3 = (cx0*k2 + cx1*k1);
	ValueType wx4 = (cx1*k2);
	
	if (UpdateWanted) {
	  ValueType Vx_suggested = wx1*H;
	      Vx_suggested = wx2*H;
	      Vx_suggested = wx3*H;
	      Vx_suggested = wx4*H;
	  setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( Vx_suggested - getG(g_Row, g_Col, g_Current, level)), level);
	  
	} else {
	  
	  if ( ComputeSurfaceVelocity ) {

	    ValueType Vx_Surface = 0.0;

	    // Compute approximation or full term?
	    if  ( ComputeSurfaceVelocity == 1 ) {
	      Vx_Surface = getG(g_Row, g_Col, g_Current, level);
	    } else {
	      Vx_Surface = wx1*H;
	      Vx_Surface = wx2*H;
	      Vx_Surface = wx3*H;
	      Vx_Surface = wx4*H;
	    };

	    setG(g_Row, g_Col, g_Vxs, Vx_Surface, level);
	  } else {
	    
	    ValueType Vx_suggested = wx1*H;
	    Vx_suggested = wx2*H;
	    Vx_suggested = wx3*H;
	    Vx_suggested = wx4*H;
	    ValueType Vx_error     = getG(g_Row, g_Col, g_Wanted, level);
	    Vx_error = (Vx_suggested - getG(g_Row, g_Col, g_Current, level)); // r = f - A(u)
	    ValueType Vx_new       = getG(g_Row, g_Col, g_Current, level) - Vx_error*constMem.Relaxv;
	    
	    setG(g_Row, g_Col, g_Current  , Vx_new, level); 
	    setG(g_Row, g_Col, g_Suggested, Vx_suggested, level);
	    setG(g_Row, g_Col, g_Error    , Vx_error    , level);
	  };
	};
	};	
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
  void g_update_vy(ValueType *g_Current, ValueType *g_Suggested,  ValueType *g_Wanted, ValueType *g_Error, ValueType *g_Vys,
		   const ValueType *g_Sxx, const ValueType *g_Syy, const ValueType *g_Sxy, 
		   const ValueType *g_Sxxm, const ValueType *g_Syym, const ValueType *g_Sxym, 
		   const ValueType *g_IceTopo, const ValueType *g_IceTopom, const ValueType *g_BedTopo, 
		   const int level, const int color, const int ComputeSurfaceVelocity, const bool UpdateWanted)   
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

       	if ( g_Row > 1 ) {
	   
	    // Compute x-cofficients in velocity position
	    ValueType Sxx = ((getG(g_Row, g_Col, g_Sxx, level)+getG(g_Row, g_Col-1, g_Sxx, level))/2.0);
	    ValueType Syy = ((getG(g_Row, g_Col, g_Syy, level)+getG(g_Row, g_Col-1, g_Syy, level))/2.0);
	    ValueType Sxy = ((getG(g_Row, g_Col, g_Sxy, level)+getG(g_Row, g_Col-1, g_Sxy, level))/2.0);

	    ValueType cx0 = 
	      (2.0 * Sxx + Syy)
	      * gradVelocity(g_Row, g_Col  , g_IceTopom, 1, 0, level)      // d(h)/dx
	      + Sxy
	      * gradVelocity(g_Row, g_Col-1, g_IceTopo , 0, 1, level);     // d(h)/dy
	    
	    ValueType cx1 = 
	      - 1.0 * gradVelocity(g_Row  , g_Col  , g_IceTopom, 1, 0, level)  // d(h)/dx
	      + 2.0 * gradVelocity(g_Row  , g_Col  , g_Sxxm    , 1, 0, level)  // d(Sxx)/dx
	      +       gradVelocity(g_Row  , g_Col  , g_Syym    , 1, 0, level)  // d(Syy)/dx
	      +       gradVelocity(g_Row  , g_Col-1, g_Sxy     , 0, 1, level); // d(Sxy)/dy
	    
	    // Compute y-cofficients in velocity position
	    ValueType cy0 = 
	      ( Sxx + 2.0 * Syy)
	      * gradVelocity(g_Row  , g_Col-1, g_IceTopo , 0, 1, level)       // d(h)/dy
	      + Sxy 
	      * gradVelocity(g_Row  , g_Col  , g_IceTopom, 1, 0, level);      // d(h)/dx
	    
	    ValueType cy1 = 
	      - 1.0 * gradVelocity(g_Row, g_Col-1, g_IceTopo, 0, 1, level)   // d(h)/dy
	      +       gradVelocity(g_Row, g_Col-1, g_Sxx    , 0, 1, level)   // d(Sxx)/dy
	      + 2.0 * gradVelocity(g_Row, g_Col-1, g_Syy    , 0, 1, level)   // d(Syy)/dy
	      +       gradVelocity(g_Row, g_Col  , g_Sxym   , 1, 0, level);  // d(Sxy)/dx
	    
	    
	    ValueType taud2 = Sxx*Sxx + Syy*Syy + Sxx*Syy + Sxy*Sxy;
	    
	    ValueType k0 = cx0*cx0 + cy0*cy0 + taud2;
	    ValueType k1 = 2.0 * (cx0*cx1 + cy0*cy1);
	    ValueType k2 = cx1*cx1 + cy1*cy1;
	    
	    ValueType H = (
			   (getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level))
			   + (getG(g_Row, g_Col-1, g_IceTopo, level) - getG(g_Row, g_Col-1, g_BedTopo, level))
			   )/2.0; 


	      // y-coefficients
	      ValueType wy1 = (cy0 * k0);
	      ValueType wy2 = (cy0*k1 + cy1*k0);
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
		//ValueType Vy_error     = ( Vy_suggested - getG(g_Row, g_Col, g_Current, level));
		ValueType Vy_error     = getG(g_Row, g_Col, g_Wanted, level) - ( Vy_suggested - getG(g_Row, g_Col, g_Current, level));
		ValueType Vy_new       = getG(g_Row, g_Col, g_Current, level) - Vy_error*constMem.Relaxv;
		
		setG(g_Row, g_Col, g_Current  , Vy_new      , level);
		setG(g_Row, g_Col, g_Suggested, Vy_suggested, level);
		setG(g_Row, g_Col, g_Error    , Vy_error    , level);

	      };

	      };

        };
      };
  #ifdef __CPU__
      }
    }
  #endif
};





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
				 const int level, const int color, const bool UpdateWanted)   
{

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING;   
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
      for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
	  if ((g_Row*g_Col % 2) == color) {
  #endif

	  // Compute x-cofficients
	  ValueType cx0 = 
	    (2.0 * getG(g_Row, g_Col, g_Sxx, level) + getG(g_Row, g_Col, g_Syy, level)) 
	    * grad(g_Row, g_Col, g_IceTopo, 1, 0, level)       // d(h)/dx
	    + getG(g_Row, g_Col, g_Sxy, level) 
	    * grad(g_Row, g_Col, g_IceTopo, 0, 1, level);      // d(h)/dy

	  ValueType cx1 = 
	    - 1.0 * grad(g_Row, g_Col, g_IceTopo, 1, 0, level)   // d(h)/dx
	    + 2.0 * grad(g_Row, g_Col, g_Sxx    , 1, 0, level)   // d(Sxx)/dx
	    +       grad(g_Row, g_Col, g_Syy    , 1, 0, level)   // d(Syy)/dx
	    +       grad(g_Row, g_Col, g_Sxy    , 0, 1, level);  // d(Sxy)/dy

	  // Compute y-cofficients
	  ValueType cy0 = 
	    (getG(g_Row, g_Col, g_Sxx, level) + 2.0 * getG(g_Row, g_Col, g_Syy, level))
	    * grad(g_Row, g_Col, g_IceTopo, 0, 1, level)       // d(h)/dy
	    + getG(g_Row, g_Col, g_Sxy, level) 
	    * grad(g_Row, g_Col, g_IceTopo, 1, 0, level);      // d(h)/dx

	  ValueType cy1 = 
	    - 1.0 * grad(g_Row, g_Col, g_IceTopo, 0, 1, level)   // d(h)/dy
	    +       grad(g_Row, g_Col, g_Sxx    , 0, 1, level)   // d(Sxx)/dy
	    + 2.0 * grad(g_Row, g_Col, g_Syy    , 0, 1, level)   // d(Syy)/dy
	    +       grad(g_Row, g_Col, g_Sxy    , 1, 0, level);  // d(Sxy)/dx
	  

	  ValueType taud2 = 
	      getG(g_Row, g_Col, g_Sxx, level) * getG(g_Row, g_Col, g_Sxx, level)
	    + getG(g_Row, g_Col, g_Sxx, level) * getG(g_Row, g_Col, g_Syy, level)
	    + getG(g_Row, g_Col, g_Syy, level) * getG(g_Row, g_Col, g_Syy, level)
	    + getG(g_Row, g_Col, g_Sxy, level) * getG(g_Row, g_Col, g_Sxy, level);
	  
	  ValueType k0 = cx0*cx0 + cy0*cy0 + taud2;
	  ValueType k1 = 2.0 * (cx0*cx1 + cy0*cy1);
	  ValueType k2 = cx1*cx1 + cy1*cy1;
	  
	  ValueType H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); 

	  if (UpdateWanted) {
	    ValueType Taue_suggested = (k0 + 0.5 * k1 * H + (k2 * H*H)/3.0);             // Used to compute residual
	    setG(g_Row, g_Col, g_Wanted, getG(g_Row, g_Col, g_Wanted, level) + ( Taue_suggested - getG(g_Row, g_Col, g_Current, level)), level);
	    
	  } else {
	    
	    // Tau_Old - Tau_New
	    ValueType Taue_suggested = (k0 + 0.5 * k1 * H + ( k2 * H*H)/3.0);             // Used to compute residual
	    //ValueType Taue_error     = ( Taue_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	    ValueType Taue_error     = getG(g_Row, g_Col, g_Wanted, level) - ( Taue_suggested - getG(g_Row, g_Col, g_Current, level));   // Current residual (r = f - A(v))
	    ValueType Taue_new       = getG(g_Row, g_Col, g_Current, level) - constMem.Relaxs*Taue_error;
	    
	    setG(g_Row, g_Col, g_Current  , Taue_new      , level); // Set new current Tau_e based on updated coefficients
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
    int g_Col = 1;
    int g_Row = 1;
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
    //    cuPrintf("Hi from g_Row %i \n",g_Row);
  #else
    for (int g_Row = PADDING+yShift; g_Row < dimensions; g_Row++) { // Changed to PADDING
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
	} else {
	  Sum += getG(g_Row, Col, g_MNew, level) * getG(g_Row, Col, g_MNew, level);
	};
      };


      // Save partial residuals vector
      g_Residual[g_Row] = Sum;
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
		     const ValueType *g_Taue, const int level, const int color)   
{

  #ifdef __GPU__
  
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  g_Row = ((color+g_Col)%2)+2*g_Row-1;

  if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level]-1;g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level]-1;g_Row++) {
  #endif

	ValueType t_Taue = getG(g_Row, g_Col, g_Taue, level);
	ValueType t_Sxx  = getG(g_Row, g_Col, g_Exx, level)/(constMem.A*t_Taue + constMem.Gamma);
	ValueType t_Syy  = getG(g_Row, g_Col, g_Eyy, level)/(constMem.A*t_Taue + constMem.Gamma);
	ValueType t_Sxy  = getG(g_Row, g_Col, g_Exy, level)/(constMem.A*t_Taue + constMem.Gamma);
	
	setG(g_Row, g_Col, g_Sxx, t_Sxx, level);
	setG(g_Row, g_Col, g_Syy, t_Syy, level);
	setG(g_Row, g_Col, g_Sxy, t_Sxy, level);
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
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING+1;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING+1; 
  
    if (  g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
#else
      for (int g_Col=PADDING+1;g_Col<constMem.XNum[level]-1;g_Col++) {
	for (int g_Row=PADDING+1;g_Row<constMem.YNum[level]-1;g_Row++) {
#endif
	  
	  ValueType term = 0.5 * (gradVelocity(g_Row  , g_Col-1, g_Vx, 0, 1, level)  // d(Vx)/dy
				 +gradVelocity(g_Row-1, g_Col  , g_Vy, 1, 0, level)  // d(Vy)/dx
				  );
	  
	  setG(g_Row, g_Col, g_Exyc, term, level);
	  
#ifndef __GPU__
      }
#endif
  };
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
			    ValueType *g_Exy,
			    ValueType *g_Ezz,
			    ValueType *g_IceTopo, ValueType* g_BedTopo,
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
	  
	  // Simple strain rate updates
	/*
	ValueType t_Exx = gradVelocity( g_Row, g_Col, g_Vx, 1, 0, level); // d(Vx)/dx
	ValueType t_Eyy = gradVelocity( g_Row, g_Col, g_Vy, 0, 1, level); // d(Vy)/dy
	ValueType t_Exy = 0.0;
	*/
	  
	  // Complete strain rate updates
	
	    ValueType H = getG(g_Row, g_Col, g_IceTopo, level) - getG(g_Row, g_Col, g_BedTopo, level); 

	    ValueType Vx  = (getG(g_Row, g_Col, g_Vx, level)  + getG(g_Row+1, g_Col, g_Vx, level))/2.0; // Mean Ice velocity in center node
	    ValueType Vxs = (getG(g_Row, g_Col, g_Vxs, level) + getG(g_Row+1, g_Col, g_Vxs, level))/2.0; // Ice surface velocity in center node
	    ValueType Vxb = (getG(g_Row, g_Col, g_Vxb, level) + getG(g_Row+1, g_Col, g_Vxb, level))/2.0; // Ice sliding velocity in center node

	    ValueType Vy  = (getG(g_Row, g_Col, g_Vy, level)  + getG(g_Row, g_Col+1, g_Vy, level))/2.0; // Mean Ice velocity in center node
	    ValueType Vys = (getG(g_Row, g_Col, g_Vys, level) + getG(g_Row, g_Col+1, g_Vys, level))/2.0; // Ice surface velocity in center node
	    ValueType Vyb = (getG(g_Row, g_Col, g_Vyb, level) + getG(g_Row, g_Col+1, g_Vyb, level))/2.0; // Ice sliding velocity in center node

	    ValueType t_Exx = gradVelocity( g_Row, g_Col, g_Vx, 1, 0, level) - (
										grad(g_Row, g_Col, g_IceTopo, 1, 0, level) * ( Vxs - Vx  )
										+ grad(g_Row, g_Col, g_BedTopo, 1, 0, level) * ( Vx  - Vxb )
										)/(H+constMem.Gamma);
	    
	    ValueType t_Eyy = gradVelocity( g_Row, g_Col, g_Vy, 0, 1, level) // d(Vy)/dy 
	    - (
	      grad(g_Row, g_Col, g_IceTopo, 0, 1, level) * ( Vys - Vy  )
	    + grad(g_Row, g_Col, g_BedTopo, 0, 1, level) * ( Vy  - Vyb )
	    )/(H+constMem.Gamma);
	    
	    ValueType t_Exy = 
	      getG(g_Row, g_Col, g_Exy, level) - (
	      grad(g_Row, g_Col, g_IceTopo, 0, 1, level)  * ( Vxs - Vx  )
	    + grad(g_Row, g_Col, g_IceTopo, 1, 0, level)  * ( Vys - Vy  )
	    + grad(g_Row, g_Col, g_BedTopo, 0, 1, level)  * ( Vx  - Vxb )
	    + grad(g_Row, g_Col, g_BedTopo, 1, 0, level)  * ( Vy  - Vyb )
	  			     )/(2.0*H + constMem.Gamma);
					       
	
	    ValueType t_Ezz = -1.0 * (t_Exx + t_Eyy);

#ifdef __CPU__
      t_Exx *= -1;
      t_Eyy *= -1;
      t_Exy *= -1;
      t_Ezz *= -1;
#endif

     
      // Transfer back to global memory
      setG(g_Row, g_Col, g_Exx, t_Exx, level);
      setG(g_Row, g_Col, g_Eyy, t_Eyy, level);
      setG(g_Row, g_Col, g_Exy, t_Exy, level);
      setG(g_Row, g_Col, g_Ezz, t_Ezz, level);
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
			   )/4.0;                                  // but a different memory space
    
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
 void setBoundaries( ValueType *g_Vx, //!< Global pointer to x-velocity in device memory
		     ValueType *g_Vy, //!< Global pointer to y-velocity in device memory
		     ValueType *g_Vxs,
		     ValueType *g_Vys,
		     ValueType *g_Sxx, 
		     ValueType* g_Syy, 
		     ValueType* g_Sxy,
		     ValueType *g_Sxxm, 
		     ValueType* g_Syym, 
		     ValueType* g_Sxym,
		     ValueType* g_Exy,
		     ValueType* g_Exyc,
		     const int SetValues,
		     int level) {

  #ifdef __GPU__
    int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
    int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

    if ( g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {
  #else
    for (int g_Col=PADDING;g_Col<constMem.XNum[level];g_Col++) {
      for (int g_Row=PADDING;g_Row<constMem.YNum[level];g_Row++) {
  #endif      

	// NOTE: Exym perhaps has to get corner nodes filled

	if (g_Col == 1 && g_Row == 1) {
	  setG(0,0,g_Sxx,getG(constMem.YNum[level]-2,constMem.XNum[level]-2,g_Sxx,level),level); // Sxx
	  setG(0,0,g_Sxxm,getG(constMem.YNum[level]-2,constMem.XNum[level]-2,g_Sxxm,level),level); // Sxx
	};

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
	
      };

      if (SetValues == 1) {
	setG(g_Row,1,g_Exyc ,getG(g_Row,constMem.XNum[level]-2,g_Exyc,level),level); // Syy
      };

      if (SetValues == 2) {
	setG(g_Row,0,g_Exy ,getG(g_Row,constMem.XNum[level]-2,g_Exy,level),level); // Syy
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
      
      }
      if (SetValues == 1) {
	setG( g_Row, constMem.XNum[level]-1, g_Exyc, getG( g_Row, 2, g_Exyc, level), level);
      };

      if (SetValues == 2) {
	setG( g_Row, constMem.XNum[level]-1, g_Exy, getG( g_Row, 1, g_Exy, level), level);
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
      
      }
      if (SetValues == 1) {
	setG( 1, g_Col, g_Exyc, getG( constMem.YNum[level]-2, g_Col, g_Exyc, level), level);
      }

      if (SetValues == 2) {
	setG( 0, g_Col, g_Exy, getG( constMem.YNum[level]-2, g_Col, g_Exy, level), level);
      }
      /*
      setG( 0, g_Col, g_Exx, getG( constMem.YNum[level]-2, g_Col, g_Exx, level), level);
      setG( 0, g_Col, g_Eyy, getG( constMem.YNum[level]-2, g_Col, g_Eyy, level), level);
      setG( 0, g_Col, g_Exy, getG( constMem.YNum[level]-2, g_Col, g_Exy, level), level);
      */
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
	
      }

      if (SetValues == 1) {
	setG( constMem.YNum[level]-1, g_Col, g_Exyc, getG( 2, g_Col, g_Exyc, level), level);
      };

      if (SetValues == 2) {
	setG( constMem.YNum[level]-1, g_Col, g_Exy, getG( 1, g_Col, g_Exy, level), level);
      };
	/*
	setG( constMem.YNum[level]-1, g_Col, g_Exx, getG( 1, g_Col, g_Exx, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Eyy, getG( 1, g_Col, g_Eyy, level), level);
	setG( constMem.YNum[level]-1, g_Col, g_Exy, getG( 1, g_Col, g_Exy, level), level);
	*/
      };
    };
  };

  };

 #ifndef __GPU__
    };
 #endif

};


#endif








/*
 * 2D STOKE FUNTION
 */


#ifdef __2DSTOKES__
__global__ void g_vx_kernel(ValueType *Vx, const ValueType *Vy, const ValueType *P, const ValueType *R, ValueType *Res, const ValueType *Etan, const ValueType *Etas, int level, int right, int color) 
{

  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 
  
  
  if ( (g_Row+g_Col) % 2 == color ) {
  if ( g_Col > 1 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

  ValueType dCx = 0.0;

   if (g_Row == 1 && constMem.BC[2] == 0) {
     dCx = getG(g_Row,g_Col,Etas,level)*cYkf[level]; 
   };
   if (g_Row == constMem.YNum[level]-2 && constMem.BC[3] == 0) {
     dCx = getG(g_Row+1,g_Col,Etas,level)*cYkf[level]; 
   };


  // Cx[i,j] = -2 * (etan[i-1,j] - etan[i-1,j-1])/(dx * dx) - (etas[i,j] + etas[i-1,j])/(dy * dy)
  ValueType Cx = -2*cXkf[level] * (getG(g_Row,g_Col,Etan,level) + getG(g_Row,g_Col-1,Etan,level)) - cYkf[level] * (getG(g_Row,g_Col,Etas,level) + getG(g_Row+1,g_Col,Etas,level)) + dCx;

  // dxx[i,j]'/dx = 2 * etan[i-1,j] * (Vx[i,j+1] - Vx(i,j)/(dx * dx)
  //                - 2 etan[i-1,j-1] * (Vx[i,j] - Vx[i,j-1])/(dx * dx)
  // Gerya Eqn. (14.48)
  ValueType dxxdx = 2*cXkf[level] * (getG(g_Row,g_Col,Etan,level) * (getG(g_Row,g_Col+1,Vx,level) - getG(g_Row,g_Col,Vx,level)) - getG(g_Row,g_Col-1,Etan,level) * (getG(g_Row,g_Col,Vx,level) - getG(g_Row,g_Col-1,Vx,level)));
  
  // dxy[i,j]/dy = etas[i,j] * ( (Vx[i+1,j] - Vx[i,j])/(dy * dy) + (Vy[i,j+1] - Vy[i,j])/(dx * dy) )
  //               - etas[i-1,j] * ( (Vx[i,j] - Vx[i-1,j])/(dy * dy) + (Vy[i-1,j+1] - Vy[i-1,j])/(dx * dy) )
  // Gerya Eqn. (14.49)
  ValueType dxydya = getG(g_Row+1,g_Col,Etas,level) * ((getG(g_Row+1,g_Col,Vx,level) - getG(g_Row,g_Col,Vx,level)) * cYkf[level] + (getG(g_Row+1,g_Col,Vy,level) - getG(g_Row+1,g_Col-1,Vy,level)) * cXykf[level] );
  ValueType dxydyb = getG(g_Row,g_Col,Etas,level) * ((getG(g_Row,g_Col,Vx,level) - getG(g_Row-1,g_Col,Vx,level)) * cYkf[level] + (getG(g_Row,g_Col,Vy,level) - getG(g_Row,g_Col-1,Vy,level)) * cXykf[level]);
  ValueType dxydy = dxydya - dxydyb;

  
  // dP[i,j]/dx = (P[i-1,j] - P[i-1,j-1])/dx
  // Gerya Eqn. (14.30)
  ValueType dPdx = (getG(g_Row,g_Col,P,level) - getG(g_Row,g_Col-1,P,level))/constMem.Dx[level];
  
  // dRx[i,j] = Rx[i,j] - dxx[i,j]'/dx - dxy[i,j]/dy + dP[i,j]/dx
  // Gerya Eqn. (14.45)
  ValueType dRx = getG(g_Row,g_Col,R,level) - dxxdx - dxydy + dPdx; 
  
  // Updating solution
  // Vx[i,j] = Vx[i,j] + (dRx[i,j] * Theta)/Cx[i,j]
  // Gerya Eqn. (14.21)
  ValueType n_Vx = getG(g_Row,g_Col,Vx,level) + dRx*constMem.Relaxs/Cx;

  // Update the right hand side og equations
   if (right) {
     setG(g_Row, g_Col, Res, dRx, level);
  } else {
     setG(g_Row,g_Col,Vx,n_Vx,level); // Write new value
  };
  };
  };
}


 ///////////////////////////////////////////////////////////////
 //          Stencil to compute new Vy in grid                //
 ///////////////////////////////////////////////////////////////
__global__ void g_vy_kernel( const ValueType *Vx, ValueType *Vy, const ValueType *P, const ValueType *R, ValueType *Res, const ValueType *Etan, const ValueType *Etas, int level, int right, int color)
 {   
  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 


  if ( (g_Row+g_Col) % 2 == color ) {
  if ( g_Row > 1 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {

   ValueType dCy = 0.0;
   
   if (g_Col == 1 && constMem.BC[0] == 0) {
     dCy = getG(g_Row,g_Col+1,Etas,level)*cXkf[level];
   };
   if (g_Col == constMem.XNum[level]-2 && constMem.BC[1] == 0) {
     dCy = getG(g_Row,g_Col,Etas,level)*cXkf[level];
   };
   
  // (14.53)
   ValueType Cy = -2*cYkf[level]*( getG(g_Row,g_Col,Etan,level) + getG(g_Row-1,g_Col,Etan,level) ) - cXkf[level]*( getG(g_Row,g_Col+1,Etas,level) + getG(g_Row,g_Col,Etas,level) ) + dCy;

  // (14.50)
  ValueType dyydya = getG(g_Row,g_Col,Etan,level) * ( getG(g_Row+1,g_Col,Vy,level) - getG(g_Row,g_Col,Vy,level) );
  ValueType dyydyb = getG(g_Row-1,g_Col,Etan,level) * (getG(g_Row,g_Col,Vy,level) - getG(g_Row-1,g_Col,Vy,level));
  ValueType dyydy = 2.0*cYkf[level]* (dyydya - dyydyb);
  
  // (14.51)
  ValueType dyxdx = getG(g_Row,g_Col+1,Etas,level) * ( cXkf[level] * (getG(g_Row,g_Col+1,Vy,level) - getG(g_Row,g_Col,Vy,level)) + cXykf[level]*( getG(g_Row,g_Col+1,Vx,level) - getG(g_Row-1,g_Col+1,Vx,level) ) )
                  - getG(g_Row,g_Col,Etas,level) * ( cXkf[level]*( getG(g_Row,g_Col,Vy,level) - getG(g_Row,g_Col-1,Vy,level) ) + cXykf[level]*( getG(g_Row,g_Col,Vx,level) - getG(g_Row-1,g_Col,Vx,level) ) );
  
  // (14.31)
  ValueType dpdy = ( getG(g_Row,g_Col,P,level) - getG(g_Row-1,g_Col,P,level) )/constMem.Dy[level];
  
  ValueType dRy = getG(g_Row,g_Col,R,level) - dyydy - dyxdx + dpdy;
  
  ValueType n_Vy = getG(g_Row,g_Col,Vy,level) + dRy*constMem.Relaxs/Cy;

  // Update the right hand side og equations
   if (right) {
     setG(g_Row, g_Col, Res, dRy, level);
  } else {
     setG(g_Row,g_Col,Vy,n_Vy,level); // Write new value
  };
  };
  };

 };

 ///////////////////////////////////////////////////////////////
 //         Stencil to compute new P in grid                  //
 ///////////////////////////////////////////////////////////////
__global__ void g_p_kernel( const ValueType *Vx, const ValueType *Vy, ValueType *P, const ValueType *R, ValueType* Res, const ValueType *Etan, const ValueType *Etas, int level, int right, int color)
{

  int g_Col = blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row = blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  if ( (g_Row+g_Col) % 2 == color ) {
    if ( g_Col > 0 && g_Row > 0 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {


   // eq. (14.32)
   ValueType dvxdx = ( getG(g_Row,g_Col+1,Vx,level) - getG(g_Row,g_Col,Vx,level) )/constMem.Dx[level]; 
   
   // eq. (14.33)
   ValueType dvydy = ( getG(g_Row+1,g_Col,Vy,level) - getG(g_Row,g_Col,Vy,level) )/constMem.Dy[level];
   
   // eq. (14.24)
   ValueType dRc = getG(g_Row,g_Col,R,level) - (dvxdx + dvydy);
   
   // (14.47)
   ValueType n_P = getG(g_Row,g_Col,P,level) + dRc*cRelaxc*getG(g_Row,g_Col,Etan,level);

  // Update the right hand side og equations
  if (right) {
    setG(g_Row, g_Col, Res, dRc, level);
  } else {
    setG(g_Row,g_Col,P,n_P,level); // Write new value
  };
    };
  };
  
}; 


////////////////////////////////////////////////////////////
//               correctPressure			  //
//                               			  //
//  DESCRIPTION:                           		  //
//  Corrects the pressure from smoother with the          //
//  pressure from the first cell                          //
//                               			  //
//  INPUT:                                                //
//  g_M   - Pointer to matrix in global                   //
//  level - Current level                                 //
//                               			  //
//  OUTPUT:                                               //
//  None                                                  //
////////////////////////////////////////////////////////////
__global__ void correctPressure( ValueType *g_M, int level, bool setZero, ValueType PNorm) {
  // Block row and col
  int bx = blockIdx.x;   int by = blockIdx.y;  // Block ID
  int tx = threadIdx.x;  int ty = threadIdx.y; // Thread ID

  int s_Col = tx+PADDING; 
  int s_Row = ty+PADDING; 

  int g_Col = bx*blockDim.x+s_Col;
  int g_Row = by*blockDim.y+s_Row; 

  if (setZero == true) {
    // Correct pressure after first cell
    ValueType  dP = PNorm-getG(1,1,g_M,level);
    
    if (g_Row == 1 && g_Col == 1) {
      dP = 0;
      //    return;
    };
    
    if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
      setG(g_Row,g_Col,g_M,getG(g_Row,g_Col,g_M,level) + dP,level);
      
    };
  } else {
    // Set P(1,1) = pnorm
    if (g_Row == 1 && g_Col == 1) {
      setG(1,1,g_M,PNorm,level);
    };

  };

};

////////////////////////////////////////////////////////////
//            Viscosity Restriction      		  //
//                               			  //
//  DESCRIPTION:                           		  //
//  Function to only do restriction on viscosity nodes.   //
//                               			  //
//  INPUT:                                                //
//  etanf   - Normal viscosity fine grid                  //
//  etanc   - Normal viscosity coarse grid                //
//  etasf   - Shear viscosity fine grid                   //
//  etasc   - Shear viscosity coarse grid                 //
//  level   - Current level (coarse grid)                 //
//                               			  //
//  OUTPUT:                                               //
//  Returned through pointers                             //
//                               			  //
////////////////////////////////////////////////////////////
__global__ void viscosity_restriction( ValueType *etanf, ValueType *etanc, ValueType *etasf, ValueType *etasc, int CoarseLevel)
{

  int g_Col =  blockIdx.x*blockDim.x+threadIdx.x+PADDING;
  int g_Row =  blockIdx.y*blockDim.y+threadIdx.y+PADDING; 

  if ( g_Row < constMem.YNum[CoarseLevel]-1 && g_Col < constMem.XNum[CoarseLevel]-1 ) {

    int Row_fs, Col_fs, Row_fn, Col_fn, Row_c, Col_c; // Position in both arrays
    ValueType xsc, ysc, xnc, ync, xsf1, ysf1, xsf2, ysf2, xnf1, ynf1, ynf2, xnf2;

    // Coarse index
    Row_c = g_Row;
    Col_c = g_Col;

    // Coarse shear position
    xsc = (Col_c - PADDING)*constMem.Dx[CoarseLevel];
    ysc = (Row_c - PADDING)*constMem.Dy[CoarseLevel];

    xnc = xsc + 0.5*constMem.Dx[CoarseLevel];
    ync = ysc + 0.5*constMem.Dy[CoarseLevel];

    // Fine index
    Col_fs = __double2int_rn(xsc/constMem.Dx[CoarseLevel-1]);
    Row_fs = __double2int_rn(ysc/constMem.Dy[CoarseLevel-1]);


    if (Row_fs < 1)
      Row_fs = 1;

    if (Row_fs > constMem.YNum[CoarseLevel-1]-2)
      Row_fs = constMem.YNum[CoarseLevel-1]-2;

    if (Col_fs < 1)
      Col_fs = 1;

    if (Col_fs > constMem.XNum[CoarseLevel-1]-2)
      Col_fs = constMem.XNum[CoarseLevel-1]-2;


    xsf1 = (Col_fs)*constMem.Dx[CoarseLevel-1]; // Shear nodes
    ysf1 = (Row_fs)*constMem.Dy[CoarseLevel-1];
    xsf2 = xsf1 + constMem.Dx[CoarseLevel-1];
    ysf2 = ysf1 + constMem.Dy[CoarseLevel-1];

    xnf1 = (Col_fs + 0.5)*constMem.Dx[CoarseLevel-1]; // Normal nodes
    ynf1 = (Row_fs + 0.5)*constMem.Dy[CoarseLevel-1];
    xnf2 = xnf1 + constMem.Dx[CoarseLevel-1];
    ynf2 = ynf1 + constMem.Dy[CoarseLevel-1];

    ValueType dRs = 0;
    ValueType dRn = 0;
    ValueType WSums = 0.0;
    ValueType WSumn = 0.0;
    
    // Compute weights
    ValueType Ws[4], Wn[4];
    
    Ws[0] = ( 1 - (xsc  - xsf1)/constMem.Dx[CoarseLevel-1] ) * ( 1 - (ysc  - ysf1)/constMem.Dy[CoarseLevel-1] ); // Upper left
    Ws[1] = ( 1 - (xsc  - xsf1)/constMem.Dx[CoarseLevel-1] ) * ( 1 - (ysf2 - ysc )/constMem.Dy[CoarseLevel-1] ); // Lower left
    Ws[2] = ( 1 - (xsf2 - xsc )/constMem.Dx[CoarseLevel-1] ) * ( 1 - (ysc  - ysf1)/constMem.Dy[CoarseLevel-1] ); // Upper right
    Ws[3] = ( 1 - (xsf2 - xsc )/constMem.Dx[CoarseLevel-1] ) * ( 1 - (ysf2 - ysc )/constMem.Dy[CoarseLevel-1] ); // Lower right

    Wn[0] = ( 1 - (xnc  - xnf1)/constMem.Dx[CoarseLevel] ) * ( 1 - (ync  - ynf1)/constMem.Dy[CoarseLevel] ); // Upper left
    Wn[1] = ( 1 - (xnc  - xnf1)/constMem.Dx[CoarseLevel] ) * ( 1 - (ynf2 - ync )/constMem.Dy[CoarseLevel] ); // Lower left
    Wn[2] = ( 1 - (xnf2 - xnc )/constMem.Dx[CoarseLevel] ) * ( 1 - (ync  - ynf1)/constMem.Dy[CoarseLevel] ); // Upper right
    Wn[3] = ( 1 - (xnf2 - xnc )/constMem.Dx[CoarseLevel] ) * ( 1 - (ynf2 - ync )/constMem.Dy[CoarseLevel] ); // Lower right

    WSums = (Ws[0]+Ws[1]+Ws[2]+Ws[3]);
    WSumn = (Wn[0]+Wn[1]+Wn[2]+Wn[3]);
    
    // Restrict shear nodes
    dRs = Ws[0] * getG(Row_fs  ,Col_fs  , etasf,CoarseLevel-1)
        + Ws[1] * getG(Row_fs+1,Col_fs  , etasf,CoarseLevel-1)
        + Ws[2] * getG(Row_fs  ,Col_fs+1, etasf,CoarseLevel-1)
        + Ws[3] * getG(Row_fs+1,Col_fs+1, etasf,CoarseLevel-1);

    // Restrict normal nodes
    dRn = Wn[0] * getG(Row_fs  ,Col_fs  , etanf,CoarseLevel-1)
        + Wn[1] * getG(Row_fs+1,Col_fs  , etanf,CoarseLevel-1)
        + Wn[2] * getG(Row_fs  ,Col_fs+1, etanf,CoarseLevel-1)
        + Wn[3] * getG(Row_fs+1,Col_fs+1, etanf,CoarseLevel-1);


    if (CoarseLevel == 2 && Col_c == 2 && Row_c == 2) {
      cuPrintf("WSumn = %e \n",WSumn);
    };

    if (WSums && WSumn) {
      /* Do NOT do this if WSum == 0! */
      setG(Row_c,Col_c,etasc,dRs/WSums,CoarseLevel);
      setG(Row_c,Col_c,etanc,dRn/WSumn,CoarseLevel);
    } else {
      cuPrintf("CUDA wanted to divide by zero in prolong function.. That is not good!");
    };

  };
};



////////////////////////////////////////////////////////////////////////////////
//                 computeRight				                      //
//                               			                      //
//  DESCRIPTION:                           		                      //
//  Computes the new right hand side of the 		                      //
//  Stokes equation.                  			                      //
//                               			                      //
//  INPUT:                                                                    //
//  Vx    - Shared x-velocity              Row   - Shared row                 //
//  Vy    - Shared y-velocity              Col   - Shared column              //
//  P     - Shared pressure                level - Current level              //
//  Rx    - Global right hand side in x    Cx    - Boundary correction in x   //
//  Ry    - Global right hand side in Y    Cy    - Boundary correction in y   //
//  Rc    - Global right hand side in P    Etan  - Normal viscosity           //
//  g_Row - Global row                     Etas  - Shear viscosity            //
//  g_Col - Global column                                                     //
//                               			                      //
//  OUTPUT:                                                                   //
//  Resx - Residual in x                                                      //
//  Resy - Residual in y                                                      //
//  Resc - Residual in p                                                      //
////////////////////////////////////////////////////////////////////////////////
__device__ void computeRight(ValueType *Vx, ValueType *Vy, ValueType *P,
	            	     ValueType *Rx, ValueType *Ry, ValueType *Rc,
	            	     ValueType *Resx, ValueType *Resy, ValueType *Resc,
			     ValueType *Etan, ValueType *Etas,
			     int Row, int Col, int g_Row, int g_Col, int level) {


  
  if ( g_Col > 1 && Row < blockDim.y+1 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 && g_Col < constMem.XNum[level]-2 ) {
    (void) vx_kernel(Vx, Vy, P, Rx, Resx, Etan, Etas, Row, Col, g_Row, g_Col, level, 1);
    // setG(g_Row, g_Col, Resx, dRx,level);
  } else {
    setG(g_Row,g_Col,Resx,0,level);
  };
  
  if ( (Row+g_Row) > 2 && Row < blockDim.y+1 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
    (void) vy_kernel(Vx, Vy, P, Ry, Resy, Etan, Etas, Row, Col, g_Row, g_Col, level, 1);
    //setG(g_Row, g_Col, Resy, dRy,level);
  } else {
     setG(g_Row,g_Col,Resy,0,level);
  };  
  
  if ( g_Col > 0 && (Row+g_Row) > 1 && Row < blockDim.y+1 && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
    (void) p_kernel(Vx, Vy, P, Rc, Resc, Etan, Etas, Row, Col, g_Row, g_Col, level, 1);
    //setG(g_Row, g_Col, Resc, dRc,level);
  } else {
    setG(g_Row,g_Col,Resc,0,level);
  }; 
  
};


/*
void SetConstMem(int* XNum, int* YNum, ValueType XDim, ValueType YDim, ValueType Relaxs, ValueType Relaxc, ValueType* dx, ValueType* dy, int* syncIter, int* BC,
			 ValueType xkf, ValueType ykf, ValueType xykf, ValueType pnorm, int CurrentLevelNum) {
    cudaMemcpyToSymbol("cXDim", &XDim, sizeof(ValueType));
    cudaMemcpyToSymbol("cYDim", &YDim, sizeof(ValueType));
    cudaMemcpyToSymbol("constMem.Relaxs", &Relaxs, sizeof(ValueType));
    cudaMemcpyToSymbol("cRelaxc", &Relaxc, sizeof(ValueType));
    cudaMemcpyToSymbol("constMem.Dx", &dx, sizeof(ValueType)*CurrentLevelNum);
    cudaMemcpyToSymbol("constMem.Dy", &dy, sizeof(ValueType)*CurrentLevelNum);
    cudaMemcpyToSymbol("constMem.IterNum", &syncIter, sizeof(int)*CurrentLevelNum); // Number of iters per level
    cudaMemcpyToSymbol("constMem.BC", &BC, sizeof(int)*4);

    cudaMemcpyToSymbol("cXkf", &xkf, sizeof(ValueType)*CurrentLevelNum);
    cudaMemcpyToSymbol("cYkf", &ykf, sizeof(ValueType)*CurrentLevelNum);
    cudaMemcpyToSymbol("cXykf", &xykf, sizeof(ValueType)*CurrentLevelNum);
    cudaMemcpyToSymbol("cPNorm", &pnorm, sizeof(ValueType)); 
    
};
*/


////////////////////////////////////////////////////////////////////////////////
//                        setBoundaries  		                      //
//                               			                      //
//  DESCRIPTION:                           		                      //
//  Sets boundary conditions on the global level                              //
//                               			                      //
//  INPUT:                                                                    //
//  Vx    - Shared x-velocity              Row   - Shared row                 //
//  Vy    - Shared y-velocity              Col   - Shared column              //
//  P     - Shared pressure                level - Current level              //
//  g_Vx  - Global x-velocity              Cx    - Boundary correction in x   //
//  g_Vy  - Global y-velocity              Cy    - Boundary correction in y   //
//   g_P  - Global pressure                Etan  - Normal viscosity           //
//  g_Row - Global row                     Etas  - Shear viscosity            //
//  g_Col - Global column                                                     //
//                               			                      //
//  OUTPUT:                                                                   //
//  None                                                                      //
////////////////////////////////////////////////////////////////////////////////
__global__ void setBoundaries( ValueType *g_Vx, ValueType *g_Vy, ValueType *g_P, int level) {

  // Block row and col
  int bx = blockIdx.x;   int by = blockIdx.y;  // Block ID
  int tx = threadIdx.x;  int ty = threadIdx.y; // Thread ID

  int s_Col = tx+PADDING; // thread’s x-index into corresponding shared memory tile (adjusted for halos)
  int s_Row = ty+PADDING; // thread’s y-index into corresponding shared memory tile (adjusted for halos)

  int g_Col = bx*blockDim.x+s_Col;
  int g_Row = by*blockDim.y+s_Row; 

  if ( g_Row < constMem.YNum[level] && g_Col < constMem.XNum[level] ) {

  //////////////////////////////////
  //        LEFT BOUNDARY         //
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
      setG(g_Row,1,g_Vx,getG(g_Row,constMem.XNum[level]-2,g_Vx,level),level); // Vx
      setG(g_Row,0,g_Vy,getG(g_Row,constMem.XNum[level]-2,g_Vy,level),level); // Vy
      setG(g_Row,1, g_P,getG(g_Row,constMem.XNum[level]-2, g_P,level),level); // P - Set to 1 or 0
    };
    
  } else {
    //////////////////////////////////
    //         RIGHT BOUNDARY       //
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
      setG( g_Row, constMem.XNum[level]-1, g_Vx, getG( g_Row, 2, g_Vx, level), level);
      setG( g_Row, constMem.XNum[level]-1, g_Vy, getG( g_Row, 1, g_Vy, level), level);
      setG( g_Row, constMem.XNum[level]-1,  g_P, getG( g_Row, 1,  g_P, level), level);
      };
    };
  };

  //////////////////////////////////
  //         UPPER BOUNDARY       //
  //////////////////////////////////
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

    } else if (constMem.BC[2] > 1 && 0) {
      // Periodic boundaries
      return;
      };
    
  } else {
    //////////////////////////////////
    //         LOWER BOUNDARY       //
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

      } else if (constMem.BC[3] > 1 && 0) {
	// Periodic boundaries
	return;
      };
    };
  };

  };

};

#endif












/*
 * DEBUG
 */

#ifdef __DEBUG__
////////////////////////////////////////////////////////////
//                   GPUDebug    			  //
//                               			  //
//  DESCRIPTION:                           		  //
//  Used to debug device functions                        //
//                               			  //
//  INPUT:                                                //
//  g_Vx   - Pointer to x velocity in global              //
//  g_Vy   - Pointer to y velocity in global              //
//  g_P    - Pointer to pressure in global                //
//  g_Rx   - Pointer to right side x in global            //
//  g_Ry   - Pointer to right side y in global            //
//  g_Rc   - Pointer to right side p in global            //
//  g_Resx - Pointer to x residual in global              //
//  g_Resy - Pointer to y residual in global              //
//  g_Resc - Pointer to pressure residual in global       //
//  g_Etan - Pointer to normal viscosity in global        //
//  g_Etas - Pointer to shear viscosity in global         //
//  level  - Current level                                //
//  Mode   - Debug mode                                   //
//           0 : Test boundary routines                   //
//           1 : Test sharedMem routines                  //
//  GhostValue - Value to set in ghost nodes              //
//                               			  //
//  OUTPUT:                                               //
//  None                                                  //
////////////////////////////////////////////////////////////
__global__ void GPUDebug( ValueType* g_Vx, ValueType* g_Vy, ValueType* g_P, // Variables
    		       	  ValueType* g_Rx, ValueType* g_Ry, ValueType* g_Rc, // Righthand side
    		       	  ValueType* g_Resx, ValueType* g_Resy, ValueType* g_Resc, // Righthand side
			  ValueType* g_Etan, ValueType* g_Etas, // Viscosity
  	                  int level, int Mode, int GhostValue)
{

  // Block row and col
  int bx = blockIdx.x;   int by = blockIdx.y;  // Block ID
  int tx = threadIdx.x;  int ty = threadIdx.y; // Thread ID

  int s_Col = tx+PADDING; // thread’s x-index into corresponding shared memory tile (adjusted for halos)
  int s_Row = ty+PADDING; // thread’s y-index into corresponding shared memory tile (adjusted for halos)

  // Compute current block row and col
  // Remember constMem.XNum and constMem.YNum are with padding
  int g_Col = bx*blockDim.x+s_Col;
  int g_Row = by*blockDim.y+s_Row; 

  ///////////////////////////
  // Declare shared memory //
  ///////////////////////////
  __shared__ ValueType s_Vx[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Vy[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType  s_P[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Etan[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Etas[BLOCK_SIZE*BLOCK_SIZE];


  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    ///////////////////////////
    // Load Global to Shared //
    ///////////////////////////   
    set(s_Row, s_Col,   s_Vx, getG(g_Row, g_Col,   g_Vx, level));
    set(s_Row, s_Col,   s_Vy, getG(g_Row, g_Col,   g_Vy, level));
    set(s_Row, s_Col,    s_P, getG(g_Row, g_Col,    g_P, level));
    set(s_Row, s_Col, s_Etas, getG(g_Row, g_Col, g_Etas, level));
    set(s_Row, s_Col, s_Etan, getG(g_Row, g_Col, g_Etan, level));
    __syncthreads();

    ///////////////////////////////////////////////
    // Load ghost nodes into shared ghost memory //
    ///////////////////////////////////////////////
    fillGhostB( s_Vx,  g_Vx,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 1, GhostValue);
    fillGhostB( s_Vy,  g_Vy,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 1, GhostValue);
    fillGhostB( s_P,    g_P,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 1, GhostValue);
    fillGhostB( s_Etan, g_Etan, s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 1, GhostValue);
    fillGhostB( s_Etas, g_Etas, s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 1, GhostValue);
    __syncthreads();



    switch (Mode) {
    case 0:
      /********************************************
       DEBUG BOUNDARY CONDITIONS
       Put value from BC to inner nodes 
       
      ********************************************/
      if ( (by == 0 && s_Row == 2) /*|| (s_Row == 1)*/ ) {
	set(s_Row,s_Col, s_Vx, get(s_Row-1,s_Col,s_Vx)-1);
      };
      
      if ( (s_Row == blockDim.y || g_Row == constMem.YNum[level]-3 ) && g_Col > 1) {
	set(s_Row,s_Col, s_Vx, get(s_Row+1,s_Col,s_Vx)-1);
      };
      
      if ( (bx == 0 && s_Col == 2 && (g_Row+s_Row) > 2) /*|| s_Col == 1*/) {
	set(s_Row,s_Col, s_Vx, get(s_Row,s_Col-1,s_Vx)-1);
      };
      
      if ( (g_Row > 1 && ( s_Col == blockDim.x || g_Col == constMem.XNum[level]-3 )) ) {
	set(s_Row,s_Col, s_Vx, get(s_Row,s_Col+1,s_Vx)-1);
      };
      
      __syncthreads();
      break;
    case 1:
      /********************************************
       DEBUG GHOST NODES
       Function: From a zero grid add closest 
                 neighbours. Test only on pressure
                 grid.
      ********************************************/
      set(s_Row, s_Col, s_P, (  get(s_Row  , s_Col  , s_P)
			      + get(s_Row  , s_Col-1, s_P)
			      + get(s_Row  , s_Col+1, s_P)
			      + get(s_Row-1, s_Col  , s_P)
			      + get(s_Row+1, s_Col  , s_P) ) );
      __syncthreads();
      break;
    case 2:
      /********************************************
       DEBUG GHOST NODES - CORNERS
       Function: From a zero grid copy ghost 
                 nodes to internal grid. Test 
                 only on pressure grid.
      ********************************************/
      // Load from local halos
      if ( s_Row == 1 && s_Col == 1 )
	set(s_Row,s_Col,s_P,get(s_Row-1,s_Col-1,s_P));

      if (s_Row == 1 && s_Col == blockDim.x)
	set(s_Row,s_Col,s_P,get(s_Row-1,s_Col+1,s_P));

      if (s_Row == blockDim.y && s_Col == 1)
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col-1,s_P));

      if (s_Row == blockDim.y && s_Col == blockDim.x)
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col+1,s_P));

      // Load from global boundaries
      if ( g_Row == constMem.YNum[level]-2 && s_Col == 1 )
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col-1,s_P));

      if ( g_Row == constMem.YNum[level]-2 && s_Col == blockDim.x )
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col+1,s_P));

      if ( s_Row == 1 && g_Col == constMem.XNum[level]-2 )
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col+1,s_P));

      if ( s_Row == blockDim.y && g_Col == constMem.XNum[level]-2 )
	set(s_Row,s_Col,s_P,get(s_Row-1,s_Col+1,s_P));

      if ( g_Row == constMem.YNum[level]-2 && g_Col == constMem.XNum[level]-2 )
	set(s_Row,s_Col,s_P,get(s_Row+1,s_Col+1,s_P));

      __syncthreads();
      Break;

    case 3:
      // Compute color
      // Each thread computes it's current position
      // and the one below it
      int Red =  (s_Col%2)+2*s_Row-1;
      int Black = ((s_Col+1)%2)+2*s_Row-1;
      
      int g_Red = blockDim.y*blockIdx.y+Red;
      int g_Black = blockDim.y*blockIdx.y+Black;

      set(Red  , s_Col, s_P, 1);
      set(Black, s_Col, s_P, 0);
      __syncthreads();
      break;
    };
    

    ///////////////////////////
    // COPY Shared to Global //
    ///////////////////////////
    setG(g_Row,g_Col,g_Vx,get(s_Row,s_Col,s_Vx),level);
    setG(g_Row,g_Col,g_Vy,get(s_Row,s_Col,s_Vy),level);
    setG(g_Row,g_Col,g_P,get(s_Row,s_Col,s_P),level);
    
  }

    
};

#endif





















#ifdef __OLD__



__global__ void smoother( ValueType* g_Vx, ValueType* g_Vy, ValueType* g_P, // Variables
    		       	  ValueType* g_Rx, ValueType* g_Ry, ValueType* g_Rc, // Righthand side
    		       	  ValueType* g_Resx, ValueType* g_Resy, ValueType* g_Resc, // Righthand side
			  ValueType* g_Etan, ValueType* g_Etas, // Viscosity
  	                  int level, bool postProces, int RedBlack)
{

  // Block row and col
  int bx = blockIdx.x;   int by = blockIdx.y;  // Block ID
  int tx = threadIdx.x;  int ty = threadIdx.y; // Thread ID

  int s_Col = tx+PADDING; // thread’s x-index into corresponding shared memory tile (adjusted for halos)
  int s_Row = ty+PADDING; // thread’s y-index into corresponding shared memory tile (adjusted for halos)

  // Compute current block row and col
  // Remember constMem.XNum and constMem.YNum are with padding
  int g_Col = bx*blockDim.x+s_Col;
  int g_Row = by*blockDim.y+s_Row; 

  
  ///////////////////////////
  // Declare shared memory //
  ///////////////////////////
  __shared__ ValueType s_Vx[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Vy[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType  s_P[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Etan[BLOCK_SIZE*BLOCK_SIZE];
  __shared__ ValueType s_Etas[BLOCK_SIZE*BLOCK_SIZE];
  

  //  if ( g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {

    /*
    ///////////////////////////
    // Load Global to Shared //
    ///////////////////////////   
    set(s_Row, s_Col,   s_Vx, getG(g_Row, g_Col,   g_Vx, level));
    set(s_Row, s_Col,   s_Vy, getG(g_Row, g_Col,   g_Vy, level));
    set(s_Row, s_Col,    s_P, getG(g_Row, g_Col,    g_P, level));
    set(s_Row, s_Col, s_Etas, getG(g_Row, g_Col, g_Etas, level));
    set(s_Row, s_Col, s_Etan, getG(g_Row, g_Col, g_Etan, level));
    __syncthreads();


    ///////////////////////////////////////////////
    // Load ghost nodes into shared ghost memory //
    ///////////////////////////////////////////////
    fillGhostB( s_Vx,  g_Vx,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 0, 0);
    fillGhostB( s_Vy,  g_Vy,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 0, 0);
    fillGhostB( s_P,    g_P,    s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 0, 0);
    fillGhostB( s_Etan, g_Etan, s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 0, 0);
    fillGhostB( s_Etas, g_Etas, s_Row, s_Col, g_Row, g_Col, 1, blockDim.x , 1, blockDim.y , level, 0, 0);
    __syncthreads();

    */
    if ( postProces == false ) {
      
      //////////////////////////////////
      //     Call RBGS smoother       //
      //////////////////////////////////
      for(int i=0;i<constMem.IterNum[level];i++) {     
 
	RBGS(g_Vx, g_Vy, g_P, 
	     g_Rx, g_Ry, g_Rc, 
	     g_Resx, g_Resy, g_Resc,
	     g_Etan, g_Etas, level, 
	     s_Row, s_Col, 
	     g_Row, g_Col, 0);

	__syncthreads();

      };
    
    ///////////////////////////
    // COPY Shared to Global //
    ///////////////////////////
      /*
    setG(g_Row,g_Col,g_Vx,get(s_Row,s_Col,s_Vx),level);
    setG(g_Row,g_Col,g_Vy,get(s_Row,s_Col,s_Vy),level);
    setG(g_Row,g_Col,g_P,get(s_Row,s_Col,s_P),level);
      
    __syncthreads();  */
  
    } else {
      
      //////////////////////////////////
      //    Compute Right hand side   //
      //////////////////////////////////
	RBGS(g_Vx, g_Vy, g_P, 
	     g_Rx, g_Ry, g_Rc, 
	     g_Resx, g_Resy, g_Resc,
	     g_Etan, g_Etas, level, 
	     s_Row, s_Col, 
	     g_Row, g_Col, 1);
	/*
      computeRight(g_Vx, g_Vy, g_P,
		   g_Rx, g_Ry, g_Rc,
		   g_Resx, g_Resy, g_Resc,
		   g_Etan, g_Etas,
		   s_Row, s_Col, 
		   g_Row, g_Col, level);*/
	__syncthreads();

    };
    __syncthreads();
    
    // };
};

/* ############################################################
   
   Red-Black Gauss-Seidel Functions

   Staggered Grid for Multigrid
   
    R------B------R 
    |      |      |
    |      |      |
    B------R------B 
    |      |      |
    |      |      |
    R------B------R 
 
   Note: Padding not shown

   ############################################################ */


__device__ void RBGS( ValueType *Vx, ValueType *Vy, ValueType *P,  // Variables
		      ValueType *Rx, ValueType *Ry, ValueType *Rc, // Righthand side
		      ValueType *Resx, ValueType *Resy, ValueType *Resc, // Righthand side
		      ValueType *Etan, ValueType *Etas,            // Viscosity
		      int level, int Row, int Col, int g_Row, int g_Col, int right)		     
{

  // Compute color
  // Each thread computes it's current position
  // and the one below it
  int Red =  (Col%2)+2*Row-1;
  int Black = ((Col+1)%2)+2*Row-1;

  int g_Red = blockDim.y*blockIdx.y+Red;
  int g_Black = blockDim.y*blockIdx.y+Black;

  // NOTE: Comment back code and change to shared memory to revert

  /////////////////////////////
  //    COMPUTE Vx Stencil   //
  /////////////////////////////

  if ( (g_Row+g_Col) % 2 == 0 ) {
  if ( g_Col > 1/* && Row < blockDim.y+1 */ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
    (void) vx_kernel(Vx, Vy, P, Rx, Resx, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
  }
  };
  __syncthreads();

  if ( (g_Row+g_Col) % 2 == 1 ) {
  if ( g_Col > 1/* && Row < blockDim.y+1 */ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
    (void) vx_kernel(Vx, Vy, P, Rx, Resx, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
  }
  };
  __syncthreads();
  /*
  if ( g_Col > 1 && Black < blockDim.y+1 && g_Black < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1 ) {
    (void) vx_kernel(Vx, Vy, P, Rx, Resx, Etan, Etas, Black, Col, g_Black, g_Col, level, right);
  };
  __syncthreads();
  */
  
  /////////////////////////////
  //    COMPUTE Vy Stencil   //
  /////////////////////////////
  if ( (g_Row+g_Col) % 2 == 0 ) {
  if ( g_Row > 1/* && Row < blockDim.y+1*/ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
   (void) vy_kernel(Vx, Vy, P, Ry, Resy, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
    //set( Red, Col, Vy, 5);
  };
  }
    __syncthreads();

  if ( (g_Row+g_Col) % 2 == 1 ) {
  if ( g_Row > 1/* && Row < blockDim.y+1*/ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
   (void) vy_kernel(Vx, Vy, P, Ry, Resy, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
    //set( Red, Col, Vy, 5);
  };
  }
    /*
  if ( (g_Row+Black) > 2 && Black < blockDim.y+1 && g_Black < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
     (void) vy_kernel(Vx, Vy, P, Ry, Resy, Etan, Etas, Black, Col, g_Black, g_Col, level, right);
    //set( Black, Col, Vy, 0);
  };
    __syncthreads();
    */
  /////////////////////////////
  //     COMPUTE P Stencil   //
  /////////////////////////////
  if ( (g_Row+g_Col) % 2 == 0 ) {
      if ( g_Col > 0 && g_Row > 0/* && Row < blockDim.y+1*/ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
    (void) p_kernel(Vx, Vy, P, Rc, Resc, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
        };
  };
   __syncthreads();

  if ( (g_Row+g_Col) % 2 == 1 ) {
      if ( g_Col > 0 && g_Row > 0/* && Row < blockDim.y+1*/ && g_Row < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
    (void) p_kernel(Vx, Vy, P, Rc, Resc, Etan, Etas, Row, Col, g_Row, g_Col, level, right);
        };
  };
   __syncthreads();
   /*
  if ( g_Col > 0 && (g_Row+Black) > 1 && Black < blockDim.y+1 && g_Black < constMem.YNum[level]-1 && g_Col < constMem.XNum[level]-1) {
    (void) p_kernel(Vx, Vy, P, Rc, Resc, Etan, Etas, Black, Col, g_Row, g_Col, level, right);
  };
  __syncthreads();
   */
}

__device__ ValueType vx_kernel(double *Vx, double *Vy, double *P, double *R, ValueType *Res, ValueType *Etan, ValueType *Etas, int j, int i, int g_Row, int g_Col, int level, int right) {

  /* ////////////////////////////////////////////////////
  // x-Stokes stencil 
  // x-Stokes equation dSIGMAxx'/dx+dSIGMAxy/dy-dP/dx=RX
  // 
  //     +----------------------+----------------------+
  //     |                      |                      |
  //     |                      |                      |
  //     |                  vx(j-1,i)                  |
  //     |                      |                      |
  //     |                      |                      |
  //     +-----vy(j,i-1)----etas(j,i)----vy(j,i)-------+
  //     |                      |                      |
  //     |                      |                      |
  // vx(j,i-1)  P(j,i-1)     vx(j,i)    P(j,i)    vx(j,i+1)
  //     |     etan(j,i-1)      |      etan(j,i)       |
  //     |                      |                      |
  //     +----vy(j+1,i-1)---etas(j+1,i)--vy(j+1,i)-----+
  //     |                      |                      |
  //     |                      |                      |
  //     |                   vx(j+1,i)                 |
  //     |                      |                      |
  //     |                      |                      |
  //     +-------------------- -+----------------------+
  /////////////////////////////////////////////////////// */

   ValueType dCx = 0.0;

   if (g_Row == 1 && constMem.BC[2] == 0) {
     dCx = getG(g_Row,g_Col,Etas,level)*cYkf[level]; 
   };
   if (g_Row == constMem.YNum[level]-2 && constMem.BC[3] == 0) {
     dCx = getG(g_Row+1,g_Col,Etas,level)*cYkf[level]; 
   };


  // Cx[i,j] = -2 * (etan[i-1,j] - etan[i-1,j-1])/(dx * dx) - (etas[i,j] + etas[i-1,j])/(dy * dy)
  ValueType Cx = -2*cXkf[level] * (getG(g_Row,g_Col,Etan,level) + getG(g_Row,g_Col-1,Etan,level)) - cYkf[level] * (getG(g_Row,g_Col,Etas,level) + getG(g_Row+1,g_Col,Etas,level)) + dCx;

  // dxx[i,j]'/dx = 2 * etan[i-1,j] * (Vx[i,j+1] - Vx(i,j)/(dx * dx)
  //                - 2 etan[i-1,j-1] * (Vx[i,j] - Vx[i,j-1])/(dx * dx)
  // Gerya Eqn. (14.48)
  ValueType dxxdx = 2*cXkf[level] * (getG(g_Row,g_Col,Etan,level) * (getG(g_Row,g_Col+1,Vx,level) - getG(g_Row,g_Col,Vx,level)) - getG(g_Row,g_Col-1,Etan,level) * (getG(g_Row,g_Col,Vx,level) - getG(g_Row,g_Col-1,Vx,level)));
  
  // dxy[i,j]/dy = etas[i,j] * ( (Vx[i+1,j] - Vx[i,j])/(dy * dy) + (Vy[i,j+1] - Vy[i,j])/(dx * dy) )
  //               - etas[i-1,j] * ( (Vx[i,j] - Vx[i-1,j])/(dy * dy) + (Vy[i-1,j+1] - Vy[i-1,j])/(dx * dy) )
  // Gerya Eqn. (14.49)
  ValueType dxydya = getG(g_Row+1,g_Col,Etas,level) * ((getG(g_Row+1,g_Col,Vx,level) - getG(g_Row,g_Col,Vx,level)) * cYkf[level] + (getG(g_Row+1,g_Col,Vy,level) - getG(g_Row+1,g_Col-1,Vy,level)) * cXykf[level] );
  ValueType dxydyb = getG(g_Row,g_Col,Etas,level) * ((getG(g_Row,g_Col,Vx,level) - getG(g_Row-1,g_Col,Vx,level)) * cYkf[level] + (getG(g_Row,g_Col,Vy,level) - getG(g_Row,g_Col-1,Vy,level)) * cXykf[level]);
  ValueType dxydy = dxydya - dxydyb;

  
  // dP[i,j]/dx = (P[i-1,j] - P[i-1,j-1])/dx
  // Gerya Eqn. (14.30)
  ValueType dPdx = (getG(g_Row,g_Col,P,level) - getG(g_Row,g_Col-1,P,level))/constMem.Dx[level];
  
  // dRx[i,j] = Rx[i,j] - dxx[i,j]'/dx - dxy[i,j]/dy + dP[i,j]/dx
  // Gerya Eqn. (14.45)
  ValueType dRx = getG(g_Row,g_Col,R,level) - dxxdx - dxydy + dPdx; 
  
  // Updating solution
  // Vx[i,j] = Vx[i,j] + (dRx[i,j] * Theta)/Cx[i,j]
  // Gerya Eqn. (14.21)
  ValueType n_Vx = getG(g_Row,g_Col,Vx,level) + dRx*constMem.Relaxs/Cx;

  // Update the right hand side og equations
   if (right) {
     setG(g_Row, g_Col, Res, dRx, level);
     
     return 0;
     //    return dRx;
  } else {
     setG(g_Row,g_Col,Vx,n_Vx,level); // Write new value
    return 0;
  };
  
}


 ///////////////////////////////////////////////////////////////
 //          Stencil to compute new Vy in grid                //
 ///////////////////////////////////////////////////////////////
__device__ ValueType vy_kernel( double *Vx, double *Vy, double *P, double *R, ValueType *Res, ValueType *Etan, ValueType *Etas, int j, int i, int g_Row, int g_Col, int level, int right)
 {

   // y-Stokes equation stensil
   //     +-------------------- -+-------vy(j-1,i)------+----------------------+    
   //     |                      |                      |                      |
   //     |                      |                      |                      |
   //     |                  vx(j-1,i)    P(j-1,i)  vx(j-1,i+1)                |    
   //     |                      |      etan(j-1,i)     |                      |
   //     |                      |                      |                      |
   //     +-----vy(j,i-1)---etas(j,i)----vy(j,i)--etas(j,i+1)----vy(j,i+1)-----+
   //     |                      |                      |                      |
   //     |                      |                      |                      |
   //     |                  vx(j,i)      P(j,i)     vx(j,i+1)                 |    
   //     |                      |       etan(j,i)      |                      |
   //     |                      |                      |                      |
   //     +----------------------+-------vy(j+1,i)------+----------------------+
   //
   // Computing Current y-Stokes residual
   // dSIGMAyy'/dy-dP/dy
   
   ValueType dCy = 0.0;
   
   if (g_Col == 1 && constMem.BC[0] == 0) {
     dCy = getG(g_Row,g_Col+1,Etas,level)*cXkf[level];
   };
   if (g_Col == constMem.XNum[level]-2 && constMem.BC[1] == 0) {
     dCy = getG(g_Row,g_Col,Etas,level)*cXkf[level];
   };
   
  // (14.53)
   ValueType Cy = -2*cYkf[level]*( getG(g_Row,g_Col,Etan,level) + getG(g_Row-1,g_Col,Etan,level) ) - cXkf[level]*( getG(g_Row,g_Col+1,Etas,level) + getG(g_Row,g_Col,Etas,level) ) + dCy;

  // (14.50)
  ValueType dyydya = getG(g_Row,g_Col,Etan,level) * ( getG(g_Row+1,g_Col,Vy,level) - getG(g_Row,g_Col,Vy,level) );
  ValueType dyydyb = getG(g_Row-1,g_Col,Etan,level) * (getG(g_Row,g_Col,Vy,level) - getG(g_Row-1,g_Col,Vy,level));
  ValueType dyydy = 2.0*cYkf[level]* (dyydya - dyydyb);
  
  // (14.51)
  ValueType dyxdx = getG(g_Row,g_Col+1,Etas,level) * ( cXkf[level] * (getG(g_Row,g_Col+1,Vy,level) - getG(g_Row,g_Col,Vy,level)) + cXykf[level]*( getG(g_Row,g_Col+1,Vx,level) - getG(g_Row-1,g_Col+1,Vx,level) ) )
                  - getG(g_Row,g_Col,Etas,level) * ( cXkf[level]*( getG(g_Row,g_Col,Vy,level) - getG(g_Row,g_Col-1,Vy,level) ) + cXykf[level]*( getG(g_Row,g_Col,Vx,level) - getG(g_Row-1,g_Col,Vx,level) ) );
  
  // (14.31)
  ValueType dpdy = ( getG(g_Row,g_Col,P,level) - getG(g_Row-1,g_Col,P,level) )/constMem.Dy[level];
  
  ValueType dRy = getG(g_Row,g_Col,R,level) - dyydy - dyxdx + dpdy;
  
  ValueType n_Vy = getG(g_Row,g_Col,Vy,level) + dRy*constMem.Relaxs/Cy;

  // Update the right hand side og equations
   if (right) {
     setG(g_Row, g_Col, Res, dRy, level);
    return 0;
  } else {
     setG(g_Row,g_Col,Vy,n_Vy,level); // Write new value
    return 0;
  };

 };

 ///////////////////////////////////////////////////////////////
 //         Stencil to compute new P in grid                  //
 ///////////////////////////////////////////////////////////////
__device__ ValueType p_kernel( double *Vx, double *Vy, double *P, double *R, ValueType* Res, ValueType *Etan, ValueType *Etas, int j, int i, int g_Row, int g_Col, int level, int right)
 {
   // eq. (14.32)
   ValueType dvxdx = ( getG(g_Row,g_Col+1,Vx,level) - getG(g_Row,g_Col,Vx,level) )/constMem.Dx[level]; 
   
   // eq. (14.33)
   ValueType dvydy = ( getG(g_Row+1,g_Col,Vy,level) - getG(g_Row,g_Col,Vy,level) )/constMem.Dy[level];
   
   // eq. (14.24)
   ValueType dRc = getG(g_Row,g_Col,R,level) - (dvxdx + dvydy);
   
   // (14.47)
   ValueType n_P = getG(g_Row,g_Col,P,level) + dRc*cRelaxc*getG(g_Row,g_Col,Etan,level);

  // Update the right hand side og equations
  if (right) {
    setG(g_Row, g_Col, Res, dRc, level);
    return 0;
    //    return dRc;
  } else {
    setG(g_Row,g_Col,P,n_P,level); // Write new value
    return 0;
  };

 }; 

#endif





// ****************************************************** //
//             CPU Version of routines                    //
// ****************************************************** //

