#include "config.h"

// MISC. UTILITY FUNCTIONS

// Error handler for CUDA GPU calls. 
//   Returns error number, filename and line number containing the error to the terminal.
//   Please refer to CUDA_Toolkit_Reference_Manual.pdf, section 4.23.3.3 enum cudaError
//   for error discription. Error enumeration starts from 0.

/**
 // Generic function to check if any cuda errors have happend
 // @param checkpoint_description A short message printed to the user
 // @param DoSync Do a cudaDeviceSynchronize before the error check
 */
void checkForCudaErrors(const char* checkpoint_description, bool DoSync)
{
#ifdef __GPU__
  //f (DoSync)
  cudaDeviceSynchronize();
  

  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    Error(1,checkpoint_description,err);
  };
#endif
}

#ifdef __GPU__

/**
 // Special case to handle backword compatabillity
 // @param checkpoint_description A short message printed to the user
 */
void checkForCudaErrors(const char* checkpoint_description)
{
  cudaError_t err = cudaGetLastError();
  if (err != cudaSuccess) {
    Error(1,checkpoint_description,err);
  };
}

Error::Error(int ErrCode, const char* ErrMessage, cudaError_t err) 
: Message(""), Code(0) 
{
  Message = ErrMessage;
  Code = ErrCode;
  
  // CUDA Errors
  fprintf(stdout,"CUDA Error: %s \n",Message);
  fprintf(stdout,"cudaError: %s \n",cudaGetErrorString(err));

  exit(1);
  
 };
#endif



/*

  ERROR CODES
  -10 : Number of levels in constant memory is out of expected parameters!
  -11 : Constant Memory is not set!
  -12 : Constant Memory is not set!

 */
Error::Error(int ErrCode, const char* ErrMessage) 
: Code(0), Message("") 
{
  Message = ErrMessage;
  Code = ErrCode;
  
  /* Handle Error */
  if (-5 < Code || Code < 0) {
    // Warning
    fprintf(stdout,"WARNING: %s\n",Message);
  } else if (Code > -30) {
    // Host Errors
    switch (Code) {
    case -10:
      fprintf(stdout, "ERROR: Number of levels in constant structure in global memory space is outside of expected parameter! (0 < Levels <= maxLevels)\n");
      break;
    case -11:
      fprintf(stdout, "ERROR: One or more constant memory arrays does not match the constant memory structure in global memory space!\n");
      break;
    case -12:
      fprintf(stdout, "ERROR: One or more single variable in constant memory does not match the constant memory structure in global memory space!\n");
      break;
    default:
      fprintf(stdout,"Custom error does not fit any predefined error codes!\n");    
    };

    if (Message != NULL)
      fprintf(stdout,"Custom Message: %s \n",Message);

  } else {
    if ( strcmp(Message,"") )
      fprintf(stdout,"MESSAGE: %s \n", Message);
  };
  
  /* -1 -> -5: Only warning */
  if (0 > Code || Code > -5) {
    exit(1);
  }
  
 };

void Error::HandleError() {


};

/* Error Handeling
 Uses standart netCDF and
 custom error codes*/
void Error::WriteError() {
  
  std::cout << "Error: foo bar" << std::endl;

};

/* Handle errors by printing an error message and exiting with a
 * non-zero status. */
//#define ERR(e) {printf("ERROR: %s\n", nc_strerror(e)); return 1;}

