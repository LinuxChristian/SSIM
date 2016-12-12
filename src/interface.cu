
/* Include configurations */
#ifndef CONFIG_H_
  #include "config.h"
  #include <string.h>
#endif

/*
class SSIM {
 
 public:
  void cpyMemToGPU (celltype cells,hptype hp,vptype vp, cornertype cp, meshtype &mesh, iproptype iprop);
  //  void cpyMemToGPU();
  void cpyMemToCPU();
  void iSOSIAGPU();
  void initialize();
};
*/


namespace SSIM {
/*!
 Used to initialize the multigrid before
 first run. 
 Values are loaded from a ASCII file and
 stores in RAM for later use. 
 */
  void initialize() {
  
    size_t TotalMemSize = sizeof(ValueType)*10000*10000; // Size of avaible memory in bytes
                                                       // Only important on GPU

    size_t TotalConstMemSize = sizeof(ValueType)*10000*10000;
    
    // Check the GPUs
    /*
      #ifdef __GPU__
      Log.initializeGPU(&TotalMemSize, &TotalConstMemSize);
      checkForCudaErrors("Initilazation of GPU!",0);
      #endif
    */
  };
  
  
  void cpyMemToGPU (celltype cells,hptype hp,vptype vp, cornertype cp, meshtype &mesh, iproptype iprop) {

  }
  
  void iSOSIAGPU () {
    
  };

}
