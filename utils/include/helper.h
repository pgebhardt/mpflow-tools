#ifndef MPFLOW_TOOLS_UTILS_HELPER_H
#define MPFLOW_TOOLS_UTILS_HELPER_H

// load measurement data from MWI
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> loadMWIMeasurement(
    std::string const filename, unsigned const frequencyIndex, bool const useReflectionParameter,
    cudaStream_t const cudaStream);

// print properties of current cuda device
void printCudaDeviceProperties();

// get basic compiler name
std::string getCompilerName();

// create proper file name for a reconstructed image
std::string getReconstructionFileName(int const argc, char* const argv[], json_value const& config,
    unsigned const iteration);

#endif
