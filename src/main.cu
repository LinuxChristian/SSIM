/* ##############################################################
    Copyright (C) 2012 Christian Brædstrup

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

//    SSIM Simulates Ice Motion (SSIM) is a program to compute dynamics of
//    glaciers using CUDA.
//
//    Contact:
//      Christian Brædstrup
//      christian.fredborg@geo.au.dk
//      Ice, Water and Sediment Group
//      Institut for Geoscience
//      University of Aarhus
//

#ifndef CONFIG_H_
#include "config.h"
#include <string.h>

typedef boost::posix_time::ptime Time;
typedef boost::posix_time::time_duration TimeDuration;

#endif

int main(int argc, char* argv[])
{
  using std::cout;
  using std::endl;
  using std::ofstream;

  Time tglobalbegin(boost::posix_time::microsec_clock::local_time());

  int next_option; // Commandline argc
  int ncid, ncerr; // NetCDF variables
  Log Log(NULL);

  ////////////////////////////////
  // Pass commandline arguments //
  ////////////////////////////////
  /* A string listing valid short options letters.  */
  const char* const short_options = "hd:o:i:l:vsL:nc";
  /* An array describing valid long options.  */
  const struct option long_options[] = {
    { "help",         0, NULL, 'h' },
    { "output",       1, NULL, 'o' },
    { "input",        1, NULL, 'i' },
    { "verbose",      0, NULL, 'v' },
    { "debug",        1, NULL, 'd' },
    { "cpu",          0, NULL, 'c' }, // (CPU or GPU)
    { "silent",       0, NULL, 's' },
    { "dry-run",      0, NULL, 'n' },
    { "log",          1, NULL, 'l' },
    { "license",      0, NULL, 'L' },
    { NULL,           0, NULL, 0   }   /* Required at end of array.  */
  };
  
 /* The name of the file to receive program output, or NULL for
     standard output.  */
  const char* input_filename = NULL;
  const char* output_filename = NULL;
  //  const char* log_filename = NULL;
  // Log.SetLogFile(NULL);

  /* Whether to display verbose messages.  */
  // int verbose = 0; 
  //int silent = 0; /* Suppress output */
  char* debug = NULL; /* DEFAULT set to NULL */
  // int GPUDMode = 0; /* Type of GPU debugging to preform. Only set if debug = d */
  int CPU = 0; // Default GPU
  int dryrun = 0; // Do a dry run
  ofstream ResidualFile;

  /* Remember the name of the program, to incorporate in messages.
     The name is stored in argv[0].  */
  //  const char* program_name = argv[0];
  Log.SetProgramName(argv[0]);
  
  do {
    next_option = getopt_long (argc, argv, short_options,
                               long_options, NULL);
    switch (next_option)
      {
      case 'h':   /* -h or --help */
	Log.print_usage (stdout, 0);
	
      case 'o':   /* -o or --output */
	output_filename = optarg;
	break;

      case 'i':   /* -i or --input */
	input_filename = optarg;
	break;
	
      case 'v':   /* -v or --verbose */
	Log.ChangeVerbose(1);
	break;

      case 'c':
	CPU = 1;
	break;

      case 'd':   /* -d or --debug */
	debug = optarg;
	break;

      case 's':   /* -s or --silent */
	Log.ChangeVerbose(0);
	break;

      case 'n':   /* -n or --dry-run */
	dryrun = 1;
	break;

      case 'l':   /* -l or --log */
	Log.SetLogFile(optarg);
	break;

      case 'L':   /* -L or --license */
	Log.print_license(stdout, 0);
	break;
	
      case '?':   /* The user specified an invalid option.  */
	Log.print_usage (stderr, 1);
	
      case -1:
	/* Check log file */
	// TOTO: Log.SetLogFile(log_filename);
	/*	if (log_filename != NULL) {
	  Log.SetLogFile(NULL);
	  OutStream = fopen(log_filename,"w");
	  };*/
	
	// Check users input
	// TODO: Rewrite line as Log.Write()
	
	/*	if (Log.IsSilent() && Log.IsVerbose() )
	  Error(-2,"Cannot be both silent and verbose!\n Use -l or --log to save output to file");
	*/
	if (input_filename == NULL)
	  Error(-3,"Cannot run without a input file!");
	
	// Set output file equal to input file
	if (output_filename == NULL) {
	  output_filename = input_filename;
	} else {
	  // Copy input file and use as base of output file
	  char call[200];
	  strcpy(call,"cp "); strcat(call,input_filename); strcat(call," "); strcat(call,output_filename);
	  Log.Write(call,"");
	};

	break;
	
      default:    /* Something else: unexpected.  */
	abort ();
      }
  }
  while (next_option != -1);

  #ifndef __GPU__
    if (!CPU) {
      std::cout << "Code is compiled with __CPU__ flag. CPU flag forced!" << std::endl;
      CPU = 1;
    };
  #endif

  if (CPU) {
    std::cout << "Switched to CPU Version! \n" << std::endl;
  } else {
    std::cout << "Switched to GPU Version! \n" << std::endl;
  };

  Log.PrintBanner(input_filename, output_filename);

  #ifdef __CPU__
  int sys_proc_num = omp_get_num_procs();
  int proc_num = 1;
  std::cout << "The system has " << sys_proc_num << " processor(s)" << std::endl;
  omp_set_num_threads(proc_num);
  std::cout << "I will use " << proc_num << " processor(s)" << std::endl;
  #endif

#ifdef __2DSTOKES__
  std::cout << "Build using: 2D Stokes Equations" << std::endl;
#elif __3DSTOKES__
  std::cout << "Build using: 3D Stokes Equations" << std::endl;
  std::cout << "Error: Not yet implemented!" << std::endl;
  exit(0);
#elif __iSOSIA__
    std::cout << "Build using: iSOSIA Equations" << std::endl;
    std::cout << "Warning: Not yet fully implemented!" << std::endl;
#else
    std::cout << "ERROR: This flag is not recgonized!" << std::endl;
    std::cout << "       Please see main.cu!" << std::endl;
    exit(0);
#endif

  /* Open the file. NC_NOWRITE tells netCDF we want read-only access
   * to the file.*/
    ncerr = nc_open(output_filename, NC_WRITE, &ncid);
    if ( ncerr )
      Error(ncerr,"");

  // Open a file to write residual
  std::string ResFileName;
  std::string FileExtension = ".data";
  ResFileName = output_filename;
  ResFileName.erase(ResFileName.length()-3, 3);
  ResidualFile.open((ResFileName+FileExtension).c_str() );

  std::cout << "\n Writing residual infomation to " << ResFileName << ".data"<< std::endl;
  
  size_t TotalMemSize = sizeof(ValueType)*10000*10000; // Size of avaible memory in bytes
                                                       // Only important on GPU

  size_t TotalConstMemSize = sizeof(ValueType)*10000*10000;

  profile prof; // Create variable for profiling

  #ifdef __GPU__
  if (!CPU) {
    Log.initializeGPU(&TotalMemSize, &TotalConstMemSize);
    checkForCudaErrors("Initilazation of GPU!",0);
  };
  #endif

  // Allocate Multigrid Structure and all levels 
  Multigrid MG(ncid, CPU, TotalMemSize, TotalConstMemSize);
  MG.ReadNetCDF(ncid);
  checkForCudaErrors("Allocating Multigrid structure!",0);

  std::cout << "LOG: Setting Constant Memory" << std::endl;
  MG.SetConstMemOnDevice();
  checkForCudaErrors("Setting constant memory!",0);

  if (dryrun) 
    //    std::cout << "This should have been a dryrun" << std::endl;
    exit(0);

  MG.AddLevels();

  Log.Write("LOG:","Multigrid and all levels created");
  fprintf(stdout,"Levels: %i \n",MG.GetCurrentLevels());

  #ifdef __DEBUG__
  /* Inform about debugging mode */
  if (debug != 0) {
    // Enter Debug mode
    MG.Debug(debug, ncid, CPU);
    
    if ( ncerr = nc_close(ncid) )
      Error(ncerr,"");
    
    Log.CloseLog();
    std::cout << "Finished with sucess \n" << std::endl;
    exit(0);
  }
  #endif
  
 
  //////////////////////////////////////////////////
  //  READ INPUT FROM NETCDF FILE TO CPU MEMORY   //
  //////////////////////////////////////////////////  
  MG.ReadInputToLevels(ncid);

 if ( Log.IsVerbose() )
   Log.Write("LOG:","Input read with sucess");

     
  /////////////////////////////////
  //      SETUP GPU MEMORY       //
  /////////////////////////////////
 // Declare constant memory - DEVICE
 if (Log.IsVerbose())
   Log.Write("LOG:","Constant memory tranfered to GPU");


  // Copy finest grid
  // Host -> Device
 for (unsigned int i = 0; i < MG.GetCurrentLevels(); i++) {
   MG.CopyLevelToDevice(i);
 }

 if (Log.IsVerbose())
   Log.Write("LOG:","All levels tranfered to GPU");
  

  /////////////////////////////////////////////
  //       RESTRICT TO SELECTED GRIDS        //
  /////////////////////////////////////////////
  // NOTE: Level 0 needs to have boundary nodes filled!
 
   #ifdef __GPU__
     cudaPrintfInit();
   #endif

  if (debug == NULL) {
     
    if (Log.IsVerbose() && MG.GetCurrentLevels() > 1)
      Log.Write("LOG:","Restriction");


    // TODO: Test this function on iSOSIA!!
    for (unsigned int level=0;level<MG.GetCurrentLevels()-1;level++) {
      // Note: iSOSIA restricts topography
      //       Stokes restricts viscosity
      //MG.FineToCoarse(level, level+1, CPU);
    };
  };

/////////////////////////////////////////////////////////
//                                                     //
//             CALL MULTIGRID FUNTION                  //
//                                                     //
/////////////////////////////////////////////////////////

  std::cout << std::endl;
  if (Log.IsVerbose())
    Log.Write("LOG:","------ Starting Multigrid ------");

  MG.SetFirstLevel( );

 for (unsigned int i = 0; i < MG.GetCurrentLevels(); i++) {
   MG.InterpIceTopographyToCorner(i);
 };

 // Get initial guess on solution
 // MG.RunFMG(prof);
 
 MG.SetCycleNum(-1);
 
 // Reset first level
 MG.SetFirstLevel( );
 int NextResUpdate = 1; // Get a velocity residual to begin with
 int LastResUpdate = -1;
 int ExpectedConvergence = 0;
 ValueType PreviousResidual = 0.0;

 Time tbegin(boost::posix_time::microsec_clock::local_time());
 Time tend(boost::posix_time::microsec_clock::local_time());
 TimeDuration dt;

 ValueType VelocityError = 1e2;
  while (MG.GetCycleNum() < MG.NumberOfCycles() &&  (VelocityError > 1e-24 || VelocityError < 1e-30) /*&& MG.GetCycleNum() < 11*/ ) 
    {

      std::cout << "\r" << "Running cycle " << MG.GetCycleNum();

      std::cout << " Current residual " << VelocityError << " Next update will be "<< NextResUpdate << " Convergence expected "<<  ExpectedConvergence;

      tbegin = boost::posix_time::microsec_clock::local_time();
      MG.RunVCycle(prof);
      // MG.RunWCycle(prof);
      checkForCudaErrors("Done with cycle", 1);
      tend = boost::posix_time::microsec_clock::local_time();
      dt = tend - tbegin;
      prof.tCycle += (double)(dt.total_milliseconds())/1000; // Decial seconds
      prof.nCycle += 1;      


      std::cout << "Next res update will be: " << NextResUpdate << std::endl;
      std::cout << "Convergence is expected at: " << ExpectedConvergence << std::endl;

      // The velocity residual is only computed if need be!
      // This is a very slow function!
      if ( NextResUpdate == MG.GetCycleNum() ||
	   MG.GetCycleNum() == ExpectedConvergence) {

	// Update velocity residual from GPU
	VelocityError = MG.VelocityResidual(prof);
	
	MG.ResidualCheck(MG.GetCycleNum(), VelocityError, 
			 NextResUpdate, LastResUpdate, PreviousResidual, ExpectedConvergence);

	MG.setResidual(VelocityError, MG.GetCycleNum());
	ResidualFile << VelocityError << "\n";
	ResidualFile.flush();
      }
      
      if (VelocityError > 1e4) {
	std::cout << "\n----------------------------\n";
	std::cout << "Did not converge! \n";
	std::cout << "----------------------------\n\n";
	break;
      };

      /* Try and reduce relaxation if residual is growing */
      if ( ( MG.getResidual(MG.GetCycleNum()-1) - MG.getResidual(MG.GetCycleNum()) ) < -1e-3 && 0 ) {
	MG.ReduceRelaxation(0.5);
      };
      
      
      if (  MG.GetIOUpdate() != 0 && (MG.GetCycleNum() % MG.GetIOUpdate() ) == 0) {
	// Do a full update of surface velocities
	MG.ComputeSurfaceVelocity(2);
  
	for(unsigned int i=MG.GetFirstLevel(); i < MG.GetCurrentLevels(); i++) {
	  // Write NetCDF to DISK
	  MG.CopyLevelToHost(i);	
	  MG.WriteLevelToNetCDF(ncid,i);
	  nc_sync(ncid);
	  
	  if (Log.IsVerbose()) {
	    std::cout << std::endl;
	    std::cout << "Max velocity residual is " << VelocityError << std::endl;
	  };
	};
      };
      

      MG.UpdateCycleNum();

      //      if ( MG.GetCycleNum() == 1)
      //	break;

  };
  
  // Do a full update of surface velocities
  MG.ComputeSurfaceVelocity(2);
  
  // Write 2D Slice of SxxDx, SyyDx, SxyDx...
  
  for(unsigned int i=MG.GetFirstLevel(); i < MG.GetCurrentLevels(); i++) {
    // Write NetCDF to DISK
    MG.CopyLevelToHost(i);	
    //    MG.WriteLevelToNetCDF(ncid,i);
    //    nc_sync(ncid); // Flush to disk
  }
  
 /* Close the file. This frees up any internal netCDF resources
  * associated with the file, and flushes any buffers. */
  ncerr = nc_close(ncid);
 if ( ncerr )
   Error(ncerr,"");

 Time tglobalend(boost::posix_time::microsec_clock::local_time());
 dt = tglobalend - tglobalbegin;
 prof.TotalRunTime = (double)(dt.total_milliseconds())/1000; // Decial seconds

 
 printf("\n\n PROFILING INFOMATION \n");
 printf("%-20s %6s %6s %9s\n","NAME","TIME","CALLS","PROCENT OF TOTAL RUNTIME");
 // prof.TotalRunTime = prof.tVelIter+prof.tStressIter;//+prof.tCycle;
 printf("%-20s %6.3f %6i %9.2f%\n","MAIN",prof.TotalRunTime,1,100.0);
 printf("%-20s %6.3f %6i %9.2f%\n",
	"V-Cycle",
	prof.tCycle,
	prof.nCycle,
	(double)(prof.tCycle/prof.TotalRunTime * 100));

 printf("%-20s %6.3f %6i %9.2f%\n",
	"Gauss-Seidel",
	prof.tGSIter,
	prof.nGSIter,
	(double)(prof.tGSIter/prof.TotalRunTime * 100));

 printf("%-20s %6.3f %6i %9.2f%\n",
	"Stress Iter",
	prof.tStressIter,
	prof.nStressIter,
	(double)(prof.tStressIter/prof.TotalRunTime * 100));

 printf("%-20s %6.3f %6i %9.2f%\n",
	"Velocity Iter",
	prof.tVelIter,
	prof.nVelIter,
	(double)(prof.tVelIter/prof.TotalRunTime * 100));
 
 printf("%-20s %6.3f %6i %9.2f%\n",
	"Residual",
	prof.tResidual,
	prof.nResidual,
	(double)(prof.tResidual/prof.TotalRunTime * 100));

 
 printf("%-20s %6.3f %6i %9.2f%\n",
	"Prolong",
	prof.tProlong,
	prof.nProlong,
	(double)(prof.tProlong/prof.TotalRunTime * 100));

 printf("%-20s %6.3f %6i %9.2f%\n",
	"Restrict",
	prof.tRestrict,
	prof.nRestrict,
	(double)(prof.tRestrict/prof.TotalRunTime * 100));

 // printf("Total recorded runtime     : %f \n\n",prof.TotalRunTime);

 /*
 printf("Total time to run V-cycle  : %f \n",prof.tCycle);
 printf("Total number of iterations : %i \n",prof.nCycle);
 if (prof.nCycle > 0) {
   printf("Time pr. cycle was         : %e \n", prof.tCycle/prof.nCycle);
   printf("Procentage of runtime      : %.2f \%\n", (double)(prof.tCycle/prof.TotalRunTime * 100));
 }
 printf("\n");

 printf("  Total time to run Stress   : %f \n",prof.tStressIter);
 printf("  Total number of iterations : %i \n",prof.nStressIter);
 if (prof.nStressIter > 0) {
   printf("  Time pr. cycle was         : %e \n", prof.tStressIter/prof.nStressIter);
   printf("  Procentage of runtime      : %.2f \%\n", (double)(prof.tStressIter/prof.tCycle * 100));
 }
 printf("\n");

 printf("Total time to run velocity : %f \n",prof.tVelIter);
 printf("Total number of iterations : %i \n",prof.nVelIter);
 if (prof.nVelIter > 0) {
   printf("Time pr. cycle was         : %e \n", prof.tVelIter/prof.nVelIter);
   printf("Procentage of runtime      : %.2f \%\n", (double)(prof.tVelIter/prof.tCycle * 100));
 }
 printf("\n");

 printf("Total time to run residual : %f \n",prof.tResidual);
 printf("Total number of iterations : %i \n",prof.nResidual);
 if (prof.nResidual > 0) {
   printf("Time pr. cycle was         : %e \n", prof.tResidual/prof.nResidual);
   printf("Procentage of runtime      : %.2f \%\n", (double)(prof.tResidual/prof.tCycle * 100));
 }
 */
 /*
 printf("\n");

 printf("Procentage Runtime:");
 printf("\n");
 */
 Log.CloseLog();

#ifdef __GPU__
  cudaPrintfEnd();
#endif
 
 ////////////////////////////////////////
 //   FREE MEMORY ON HOST AND DEVICE   //
 ////////////////////////////////////////
 /*
 for (int i=0; i < MG.GetCurrentLevels(); i++)
   MG.RemoveLevel();
 */

 ResidualFile.close();
 std::cout << "Finished with sucess \n" << std::endl;
 return 0;
}

