#include <mpflow/mpflow.h>
#include <distmesh/distmesh.h>
#include <sys/stat.h>
#include "stringtools/all.hpp"
#include "helper.h"

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
