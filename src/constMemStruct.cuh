
#ifndef __CONSTMEMSTRUCT_H__
#define __CONSTMEMSTRUCT_H__
#ifdef __2DSTOKES__
 struct ConstantMem {
 int XNum[maxLevel];
 int YNum[maxLevel];
 ValueType XDim;
 ValueType YDim;
 ValueType Dx[maxLevel];
 ValueType Dy[maxLevel];
 int BC[4]; // Boundary condition - 0: Free slip 1: No slip 2: Periodic
 int IterNum[maxLevel]; // Current iternum at level kc
 ValueType Relaxs;
 ValueType Relaxc;
 ValueType Xkf[maxLevel];
 ValueType Ykf[maxLevel];
 ValueType Xykf[maxLevel];
 ValueType PNorm; // Pressure to correct cell
} ConstantMem;
#endif

#ifdef __iSOSIA__
struct ConstantMem {
  int XNum[maxLevel];
  int YNum[maxLevel];
  ValueType XDim;
  ValueType YDim;
  ValueType Dx[maxLevel];
  ValueType Dy[maxLevel];
  int BC[4]; // Boundary condition - 0: Free slip 1: No slip 2: Periodic
  int IterNum[maxLevel]; // Current iternum at level kc
  ValueType Relaxs;
  ValueType Gamma;
  ValueType A;
  ValueType Relaxv;
  ValueType vKoef;
  ValueType sKoef;
  int       LevelNum; // Number of multigrid levels
  ValueType maxb;
  ValueType maxs;

  // Add sliding coefficients
  ValueType C;
  ValueType Cs;
  ValueType minbeta;
  ValueType maxsliding;
  ValueType L0;
  ValueType Relaxvb;

  int mtype;
  double lrate;
  double T0;
  double maxacc;
  double accgrad;
  double ablgrad;
  double maxabla;
  double latentheat;
  double avaslope;
  double maxslope;
  double g;
  
  double rho_w;
  double rho_i;
  double ks; // Sheet hydralic conductivity
  double kc; // Channel turbulent flow coefficient
  double Relaxh; // Relax for hydrology
  double lambdac; // incipient channel width
  double hc; // critical layer depth
  double gamma_h; 
  double GeoFlux;
  double hr;
  double lr;

  int dosliding;
  int slidingmode;
};
#endif


#endif
