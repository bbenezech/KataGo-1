#ifdef USE_OPENCL_BACKEND

#include "../neuralnet/opencltuningprogress.h"

thread_local OpenCLTuningProgress* OpenCLTuningProgress::current_ = nullptr;

#endif
