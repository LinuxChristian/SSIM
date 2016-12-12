
 /////////////////////////////////////////////////////////
 //            PRINT GENERAL INFOMATION                 //
 /////////////////////////////////////////////////////////
#include "config.h"
using namespace std;

Log::Log(FILE* Stream)
:LogStream(Stream), log_filename(NULL), program_name(NULL), Verbose(0), Silent(0)
  {
    //    LogStream = Stream;
  };



void Log::SetLogFile(const char* LogFile) {

  //LogFile.open(Filename);
  //  ostream FileStream(&LogFile);
  LogStream = fopen(LogFile,"w");
  
  if (LogStream == NULL)
    Error(-1,"Could not open log file");
	  
  Write("Log file set to",LogFile);
};

void Log::SetProgramName(const char* Name) {
    program_name = Name;
};
/*
void Log::operator<<(ofstream & os) {
  if ( IsSilent() )
    cout << os;

  if (LogFile != NULL)
    LogFile << os;

};
*/
void Log::Write(const char* LogReport, const char* LogOption) {
  
  WriteLog(LogReport, LogOption);
  WriteScreen(LogReport, LogOption);
};

void Log::CloseLog() {
  if (LogStream != NULL) {
    if ( fflush(LogStream) )
      Error(0,"Could not write buffer to log before closing");
	
    if ( fclose(LogStream) )
      Error(0,"Could not write buffer to log before closing");	    
  };
};

void Log::WriteLog(const char* LogReport, const char* LogOption ) {
  if (LogStream != NULL)
    fprintf(LogStream,"%s %s.\n", LogReport, LogOption);
};

void Log::WriteScreen(const char* LogReport, const char* LogOption) {
  if ( IsSilent() )
    fprintf(stdout,"%s %s.\n", LogReport, LogOption);
};

void Log::InitialSetup(MultigridStruct* MGS, ConstantMem * ConstMem ) {

   fprintf(LogStream,"INITIAL SETUP\n");
   #ifdef __2DSTOKES__
   fprintf(LogStream,"Relaxs %.2f Relaxc %.2f \n",ConstMem->Relaxs, ConstMem->Relaxc);
   fprintf(LogStream,"vKoef %.2f pKoef %.2f \n",ConstMem->vKoef,ConstMem->pKoef);
   fprintf(LogStream,"pnorm %.2f \n",ConstMem->pnorm);
   #endif
   fprintf(LogStream,"XDim %.2f YDim %.2f \n",ConstMem->XDim,ConstMem->YDim);
   fprintf(LogStream,"Number of levels %i \n",MGS->RequestedLevels);
   fprintf(LogStream,"Multigrid cycles %i \n",MGS->NumberOfCycles);
   fprintf(LogStream,"Output Updates %i \n",MGS->IOUpdate);
   fprintf(LogStream,"Start Level %i \n",MGS->startLevel);
   fprintf(LogStream,"Level Iterations ");
   
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream," %i ",MGS->IterationsOnLevel[i]);
   };
   fprintf(LogStream,"\nIterations between syncs ");
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream," %i ",ConstMem->IterNum[i]);
   };

   fprintf(LogStream,"\nBlock Sizes ");
   fprintf(LogStream," %i ", MGS->threadsPerBlock[0]);

   fprintf(LogStream,"\nGrid Sizes ");
   fprintf(LogStream," %i ", MGS->numBlock[0]);

   fprintf(LogStream,"\nxNum ");
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream," %i ",ConstMem->XNum[i]-2);
   };
   fprintf(LogStream,"\nyNum ");
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream," %i ",ConstMem->YNum[i]-2);
   };
   fprintf(LogStream,"\ndx ");
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream, " %.2f ",ConstMem->Dx[i]);
   };
   fprintf(LogStream,"\ndy ");
   for (int i=0;i<MGS->RequestedLevels;i++) {
     fprintf(LogStream," %.2f ",ConstMem->Dy[i]);
   };
   fprintf(LogStream,"\nBoundaires ");
   for (int i=0;i<=3;i++) {
     fprintf(LogStream," %i ",ConstMem->BC[i]);
   };
   
   fprintf(LogStream, "\n////////////// RUNNING (%i,%i) threads per block on (%i,%i) grid  /////////////\n", MGS->threadsPerBlock[0], MGS->threadsPerBlock[1], MGS->numBlock[0], MGS->numBlock[1]);

};

/* Prints usage information for this program to STREAM (typically
   stdout or stderr), and exit the program with EXIT_CODE.  Does not
   return.  */

void Log::print_usage (FILE* stream, int exit_code)
{
  fprintf (stream, "Usage:  %s options [ inputfile ... ]\n", program_name);
  fprintf (stream,
           "  -h  --help             Display this usage information.\n"
           "  -o  --output filename  Write output to file.\n"
           "  -i  --input filename   Input file to read.\n"
           "  -v  --verbose          Print verbose messages.\n"
           "  -s  --silent           Suppress output.\n"
	   "  -c  --cpu              Run solver on CPU.\n"
           "  -d  --debug type       Debug mode.\n"
	   "                         Type i - Test restriction/prolongation rutines\n"
	   "                         Type l - Test level rutines\n"
	   "                         Type b - Test boundary rutines\n"
	   "                         Type m - Test multigrid rutines\n"
	   "                         Type d# - Test device functions. # is the mode to use\n"
           "  -l  --log              Send output to log file.\n"
           "  -L  --license          Print license infomation.\n");
  exit (exit_code);
}


void Log::print_license (FILE* stream, int exit_code)
{
  fprintf (stream,
	   "/* ##############################################################        \n"
	   "   Copyright (C) 2012 Christian Brædstrup                                \n"
	   "                                                                         \n"
	   "   This program is free software: you can redistribute it and/or modify  \n"
	   "   it under the terms of the GNU General Public License as published by  \n"
	   "   the Free Software Foundation, either version 3 of the License, or     \n"
	   "   (at your option) any later version.                                   \n"
	   "                                                                         \n"
	   "   This program is distributed in the hope that it will be useful,       \n"
	   "   but WITHOUT ANY WARRANTY; without even the implied warranty of        \n"
	   "   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the         \n"
	   "   GNU General Public License for more details.                          \n"
	   "                                                                         \n"   
	   "   You should have received a copy of the GNU General Public License     \n"
	   "   along with this program.  If not, see <http://www.gnu.org/licenses/>. \n"
	   "   #################################################################     \n");
  exit (exit_code);
}

void Log::PrintBanner(const char* InputFilename, const char* OutputFilename) {

  if (LogStream != NULL) {
    fprintf(LogStream,"///////////////////////////\n");
    fprintf(LogStream,"  _____ _____ _____ __  __ \n");
    fprintf(LogStream," / ____/ ____|_   _|  \\/  |\n");
    fprintf(LogStream,"| (___| (___   | | | \\  / |\n");
    fprintf(LogStream," \\___ \\\\___ \\  | | | |\\/| |\n");
    fprintf(LogStream," ____) |___) |_| |_| |  | |\n");
    fprintf(LogStream,"|_____/_____/|_____|_|  |_|\n");
    fprintf(LogStream,"                           \n");
    fprintf(LogStream,"///////////////////////////\n");

    fprintf(LogStream,"Input datafile: %s \nOutput datafile: %s \n",InputFilename,OutputFilename);
  }

  if (!IsSilent() ) {
    fprintf(stdout,"///////////////////////////\n");
    fprintf(stdout,"  _____ _____ _____ __  __ \n");
    fprintf(stdout," / ____/ ____|_   _|  \\/  |\n");
    fprintf(stdout,"| (___| (___   | | | \\  / |\n");
    fprintf(stdout," \\___ \\\\___ \\  | | | |\\/| |\n");
    fprintf(stdout," ____) |___) |_| |_| |  | |\n");
    fprintf(stdout,"|_____/_____/|_____|_|  |_|\n");
    fprintf(stdout,"                           \n");
    fprintf(stdout,"///////////////////////////\n");

    fprintf(stdout,"Input datafile: %s \nOutput datafile: %s \n",InputFilename,OutputFilename);
  }
};

bool Log::IsVerbose() {
  return (Verbose > 0) ? true : false;
};

bool Log::IsSilent() {
  return IsVerbose();
};

void Log::ChangeVerbose(int level) {
  Verbose = level;
};


// Wrapper function for initializing the CUDA components.
// Called from main.cpp
//extern "C"
void Log::initializeGPU(size_t* TotalGlobalMem, size_t* TotalConstMem)
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
    Write("\nERROR:","No CUDA-enabled devices availible. Bye.\n");
    exit(EXIT_FAILURE);
  } else if (devicecount == 1) {
    Write("\nSystem contains 1 CUDA compatible device.\n","");
  } else {
    if (LogStream != NULL)
      fprintf(LogStream,"\nSystem contains %i CUDA compatible devices.\n",devicecount);
    if ( !IsSilent() )
      fprintf(stdout,"\nSystem contains %i CUDA compatible devices.\n",devicecount);
  }
  
    cudaGetDeviceProperties(&prop, cudadevice);
    cudaDriverGetVersion(&cudaDriverVersion);
    cudaRuntimeGetVersion(&cudaRuntimeVersion);
    checkForCudaErrors("Initializing GPU!");

    if (LogStream != NULL) {
      fprintf(LogStream,"Using CUDA device ID: %i \n",(cudadevice));
      fprintf(LogStream,"  - Name: %s, compute capability: %i.%i.\n",prop.name,prop.major,prop.minor);
      fprintf(LogStream,"  - CUDA Driver version: %i.%i, runtime version %i.%i\n",cudaDriverVersion/1000,cudaDriverVersion%100,cudaRuntimeVersion/1000,cudaRuntimeVersion%100);
      fprintf(LogStream,"  - Max threads pr. block in x: %i, Max block size in x: %i \n\n",prop.maxThreadsDim[0], prop.maxGridSize[0]);
    }

    if ( !IsSilent() ) {
      fprintf(stdout,"Using CUDA device ID: %i \n",(cudadevice));
      fprintf(stdout,"  - Name: %s, compute capability: %i.%i.\n",prop.name,prop.major,prop.minor);
      fprintf(stdout,"  - CUDA Driver version: %i.%i, runtime version %i.%i\n",cudaDriverVersion/1000,cudaDriverVersion%100,cudaRuntimeVersion/1000,cudaRuntimeVersion%100);
      fprintf(stdout,"  - Max threads pr. block in x: %i, Max block size in x: %i \n\n",prop.maxThreadsDim[0], prop.maxGridSize[0]);
    }
    // Comment following line when using a system only containing exclusive mode GPUs
    cudaChooseDevice(&cudadevice, &prop);
    checkForCudaErrors("Initializing GPU!");
    (*TotalGlobalMem) = prop.totalGlobalMem;
    (*TotalConstMem) = prop.totalConstMem;

#else
    fprintf(stdout, "Code is not compiled for GPU!");
    (*TotalGlobalMem) = 0.0;
    (*TotalConstMem) = 0.0; 

#endif

};


void msg (FILE* stream, const char* str, int type, int exit_code)
{

  if (type == 0) {
    /* ERRORS */
    /* Should always send to stdout */
    fprintf(stream, "ERROR: ");
    fprintf(stream, str);
    fprintf(stream, "\n");
    //    if (exit_code > 0)
      exit(exit_code);
  } else if (type == 1) {
    /* MESSAGES */
    fprintf(stream, str);
    fprintf(stream, "\n");
  };

};
