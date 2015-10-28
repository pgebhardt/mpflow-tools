#include "json.h"
#include <mpflow/mpflow.h>
#include <distmesh/distmesh.h>
#include <sys/stat.h>
#include <ctime>
#include "stringtools/all.hpp"
#include "helper.h"

using namespace mpFlow;

template <class dataType>
std::shared_ptr<numeric::Matrix<dataType>> loadMWIMeasurement(
    std::string const filename, unsigned const frequencyIndex, bool const useReflectionParameter, cudaStream_t const cudaStream) {
    return nullptr;
}

template std::shared_ptr<numeric::Matrix<float>> loadMWIMeasurement<float>(
    std::string const, unsigned const, bool const, cudaStream_t const);
template std::shared_ptr<numeric::Matrix<double>> loadMWIMeasurement<double>(
    std::string const, unsigned const, bool const, cudaStream_t const);

template <>
std::shared_ptr<numeric::Matrix<thrust::complex<float>>> loadMWIMeasurement(
    std::string const filename, unsigned const frequencyIndex, bool const useReflectionParameter, cudaStream_t const cudaStream) {
    // load double representation
    auto const floatMatrix = numeric::Matrix<float>::loadtxt(filename, cudaStream, ',');

    // convert to complex array
    unsigned const dim = sqrt((floatMatrix->cols - 1) / 2);
    Eigen::ArrayXXcf measurement(dim, dim);
    for (unsigned row = 0; row < measurement.rows(); ++row)
    for (unsigned col = 0; col < measurement.cols(); ++col) {
        measurement(row, col) = std::complex<float>(
            (*floatMatrix)(frequencyIndex, row * dim * 2 + col * 2 + 1),
            (*floatMatrix)(frequencyIndex, row * dim * 2 + col * 2 + 2));
    }

    // convert scattering parameter to field data
    double const fc = (constants::c0 / std::sqrt(3.1)) / (2.0 * 0.152);
    double const Z0 = std::sqrt(constants::mu0 / (constants::epsilon0 * 3.1));
    double const Zw = Z0 / std::sqrt(1.0 - math::square(fc / ((*floatMatrix)(frequencyIndex, 0) * 1e9)));
    
    Eigen::ArrayXXcf const fields = (Eigen::MatrixXcf::Identity(dim, dim).array() + measurement) * Zw / 0.016 *
        (-(useReflectionParameter ? 2.0 : 1.0) * Eigen::MatrixXcf::Identity(dim, dim).array() + Eigen::ArrayXXcf::Ones(dim, dim));

    return numeric::Matrix<thrust::complex<float>>::fromEigen(fields, cudaStream);
}

template <>
std::shared_ptr<numeric::Matrix<thrust::complex<double>>> loadMWIMeasurement(
    std::string const filename, unsigned const frequencyIndex, bool const useReflectionParameter, cudaStream_t const cudaStream) {
    // load double representation
    auto const floatMatrix = numeric::Matrix<double>::loadtxt(filename, cudaStream, ',');

    // convert to complex array
    unsigned const dim = sqrt((floatMatrix->cols - 1) / 2);
    Eigen::ArrayXXcd measurement(dim, dim);
    for (unsigned row = 0; row < measurement.rows(); ++row)
    for (unsigned col = 0; col < measurement.cols(); ++col) {
        measurement(row, col) = std::complex<double>(
            (*floatMatrix)(frequencyIndex, row * dim * 2 + col * 2 + 1),
            (*floatMatrix)(frequencyIndex, row * dim * 2 + col * 2 + 2));
    }

    // convert scattering parameter to field data
    double const fc = (constants::c0 / std::sqrt(3.1)) / (2.0 * 0.152);
    double const Z0 = std::sqrt(constants::mu0 / (constants::epsilon0 * 3.1));
    double const Zw = Z0 / std::sqrt(1.0 - math::square(fc / ((*floatMatrix)(frequencyIndex, 0) * 1e9)));
    
    Eigen::ArrayXXcd const fields = (Eigen::MatrixXcd::Identity(dim, dim).array() + measurement) * Zw / 0.016 *
        (-(useReflectionParameter ? 2.0 : 1.0) * Eigen::MatrixXcd::Identity(dim, dim).array() + Eigen::ArrayXXcd::Ones(dim, dim));

    return numeric::Matrix<thrust::complex<double>>::fromEigen(fields, cudaStream);
}

// Beginning of GPU Architecture definitions
inline int _ConvertSMVer2Cores(int major, int minor)
{
    // Defines for GPU Architecture types (using the SM version to determine the # of cores per SM
    typedef struct
    {
        int SM; // 0xMm (hexidecimal notation), M = SM Major version, and m = SM minor version
        int Cores;
    } sSMtoCores;

    sSMtoCores nGpuArchCoresPerSM[] =
    {
        { 0x20, 32 }, // Fermi Generation (SM 2.0) GF100 class
        { 0x21, 48 }, // Fermi Generation (SM 2.1) GF10x class
        { 0x30, 192}, // Kepler Generation (SM 3.0) GK10x class
        { 0x32, 192}, // Kepler Generation (SM 3.2) GK10x class
        { 0x35, 192}, // Kepler Generation (SM 3.5) GK11x class
        { 0x37, 192}, // Kepler Generation (SM 3.7) GK21x class
        { 0x50, 128}, // Maxwell Generation (SM 5.0) GM10x class
        { 0x52, 128}, // Maxwell Generation (SM 5.2) GM20x class
        {   -1, -1 }
    };

    int index = 0;

    while (nGpuArchCoresPerSM[index].SM != -1)
    {
        if (nGpuArchCoresPerSM[index].SM == ((major << 4) + minor))
        {
            return nGpuArchCoresPerSM[index].Cores;
        }

        index++;
    }

    // If we don't find the values, we default use the previous one to run properly
    printf("MapSMtoCores for SM %d.%d is undefined.  Default to use %d Cores/SM\n", major, minor, nGpuArchCoresPerSM[index-1].Cores);
    return nGpuArchCoresPerSM[index-1].Cores;
}
// end of GPU Architecture definitions

// print properties of current cuda device
void printCudaDeviceProperties() {
    // get index of current device
    int device = 0;
    cudaGetDevice(&device);

    // query device info
    cudaDeviceProp deviceProperties;
    cudaGetDeviceProperties(&deviceProperties, device);

    // get driver and runtime version
    int driverVersion = 0, runtimeVersion = 0;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    // print most important information about GPU
    str::print("Device Name:", deviceProperties.name);

    str::print(str::format("CUDA Driver Version / Runtime Version: %d.%d / %d.%d")
        (driverVersion / 1000, (driverVersion % 100) / 10, runtimeVersion / 1000, (runtimeVersion % 100) / 10));
    str::print(str::format("CUDA Device Capabilities: %d.%d")
        (deviceProperties.major, deviceProperties.minor));

    str::print(str::format("(%2d) Multiprocessors, (%3d) CUDA Cores/MP: %d CUDA Cores")
        (deviceProperties.multiProcessorCount, _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor),
        _ConvertSMVer2Cores(deviceProperties.major, deviceProperties.minor) * deviceProperties.multiProcessorCount));
    str::print(str::format("Total amount of global memory: %.0f MBytes (%llu bytes)")
        ((float)deviceProperties.totalGlobalMem/1048576.0f, (unsigned long long)deviceProperties.totalGlobalMem));
    str::print(str::format("GPU Clock rate: %.0f MHz (%0.2f GHz)")
        (deviceProperties.clockRate * 1e-3, deviceProperties.clockRate * 1e-6));

    str::print(str::format("Memory Clock rate: %.0f MHz")
        (deviceProperties.memoryClockRate * 1e-3));
    str::print(str::format("Memory Bus width: %d-bit")
        (deviceProperties.memoryBusWidth));
}

// detects compiler name and version and returns a string representation
std::string getCompilerName() {
#if defined(__clang__)
    return str::format("clang %s")(__clang_version__);
#elif defined(__GNUC__)
    return str::format("gcc %s")(__VERSION__);
#else
    return "<unknown>";
#endif
}

// create a string containing current date
std::string getCurrentDate() {
    // get current time
    time_t rawtime;
    time(&rawtime);

    // extract time info
    struct tm* timeinfo = localtime(&rawtime);

    // parse time info to string
    char buffer[15];
    strftime(buffer, 15, "%y%m%d", timeinfo);

    return std::string(buffer);
}

// extract raw filename without extension and path from given path
std::string getRawFilename(std::string const& path) {
    std::string filename = path;

    // get rid of complete path
    auto pos = path.rfind('/', path.length());
    if (pos != std::string::npos) {
        filename = path.substr(pos + 1, path.length() - pos);
    }

    // get rid of file extension
    pos = filename.rfind('.', filename.length());
    if (pos != std::string::npos) {
        filename = filename.substr(0, pos);
    }

    return filename;
}

// create proper file name for a reconstructed image
std::string getReconstructionFileName(int const argc, char* const argv[], json_value const& config,
    unsigned const iteration) {
    if (config["model"]["mwi"].type == json_none) {
        return str::format("RECON%s_%s_%s_%s_%s_RF%.0e_%s_%02dSteps_%02d.txt")(
            getCurrentDate(),
            getRawFilename(argc == 3 ? argv[2] : argv[3]),
            getRawFilename(argc == 3 ? "noRef" : argv[2]),
            config["model"]["source"]["type"].type == json_string ? std::string(config["model"]["source"]["type"]) : "current",
            config["model"]["numericType"].type == json_string ? std::string(config["model"]["numericType"]) : "real",
            config["solver"]["regularizationFactor"].u.dbl,
            std::string(config["solver"]["regularizationType"]),
            std::max(1l, config["solver"]["steps"].u.integer), iteration + 1);
    }
    else {
        // parse command line arguments
        bool const useReflectionParameter = argc > 5 ? (atoi(argv[5]) < 1 ? false : true) : true;

        auto const material = mpFlow::jsonHelper::parseNumericValue<thrust::complex<double>>(config["model"]["material"]);
        auto const frequency = config["model"]["mwi"]["frequency"].u.dbl * 1e-9;

        return str::format("R%s_%s_%s_ref_r%.0f_%.0f_i%.0f_%.0f_%.0fG%.0f_%s_RF%.0e_%s_%02dSt_%02d.txt")(
            getCurrentDate(),
            getRawFilename(argv[3]),
            getRawFilename(argv[2]),
            floor(material.real()), (material.real() - floor(material.real())) * 1e1,
            floor(material.imag()), (material.imag() - floor(material.imag())) * 1e1,
            floor(frequency), (frequency - floor(frequency)) * 1e2,
            useReflectionParameter ? "wR" : "nR",
            config["solver"]["regularizationFactor"].u.dbl,
            std::string(config["solver"]["regularizationType"]),
            std::max(1l, config["solver"]["steps"].u.integer), iteration + 1);
    }
}
