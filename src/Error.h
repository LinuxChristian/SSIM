
class Error {

 public:
  Error(int Code = 0, const char* Message = "");
  #ifdef __GPU__
  Error(int ErrCode, const char* ErrMessage, cudaError_t err);
  #endif
  ~Error( ) {};
  //  void Set(int code, const char* Message);
  //  void CheckForErrors();

 private:
  void WriteError();
  void HandleError();
  int Code;
  const char* Message;
  
};
