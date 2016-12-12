
#ifndef _LOG_
#define _LOG_

class Log {
 public:
  //  void operator<<(std::ofstream & os);
  Log(FILE* Stream);
  void InitialSetup(MultigridStruct* MGS, ConstantMem * ConstMem);
  void SetLogFile(const char* LogFile);
  void initializeGPU(size_t* TotalGlobalMem, size_t* TotalConstMem);
  void Write(const char* LogReport, const char* LogOption);
  void WriteLog(const char* LogReport, const char* LogOption );
  void WriteScreen(const char* LogReport, const char* LogOption);
  void CloseLog();
  void SetProgramName(const char* Name);
  void print_license (FILE* stream, int exit_code);
  void print_usage (FILE* stream, int exit_code);
  void PrintBanner(const char* InputFilename, const char* OutputFilename);
  bool IsVerbose();
  bool IsSilent();
  void ChangeVerbose(int level);

 private:
  FILE* LogStream; /* Stream to write log info */
  //std::ofstream LogFile;
  const char* log_filename;
  const char* program_name;
  int Verbose; /* Verbose level:
	        * 0 - No logging
		* 1 - Standart */
  bool Silent; /* Suppress output */

};

#endif
