#include <mpflow/mpflow.h>
#include <distmesh/distmesh.h>
#include <sys/stat.h>
#include "stringtools/all.hpp"
#include "json.h"
#include "helper.h"

// helper function to create an mpflow matrix from an json array
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> matrixFromJsonArray(
    json_value const& array, cudaStream_t const cudaStream) {
    // check type of json value
    if (array.type != json_array) {
        return nullptr;
    }
    
    // exctract sizes
    unsigned const rows = array.u.array.length;
    unsigned const cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create matrix
    auto matrix = std::make_shared<mpFlow::numeric::Matrix<dataType>>(rows, cols, cudaStream);

    // exctract values
    if (array[0].type != json_array) {
        for (unsigned row = 0; row < matrix->rows; ++row) {
            (*matrix)(row, 0) = array[row].u.dbl;
        }
    }
    else {
        for (unsigned row = 0; row < matrix->rows; ++row)
        for (unsigned col = 0; col < matrix->cols; ++col) {
            (*matrix)(row, col) = array[row][col].u.dbl;
        }
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
}

// creates eigen array from an json array
template <class type>
Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> eigenFromJsonArray(
    json_value const& array) {
    // check type of json value
    if (array.type != json_array) {
        return Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic>();
    }
         
    // exctract sizes
    unsigned const rows = array.u.array.length;
    unsigned const cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create array
    Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> eigenArray(rows, cols);

    // extract values
    if (array[0].type != json_array) {
        for (unsigned row = 0; row < eigenArray.rows(); ++row) {
            eigenArray(row, 0) = array[row].u.dbl;
        }
    }
    else {
        for (unsigned row = 0; row < eigenArray.rows(); ++row)
        for (unsigned col = 0; col < eigenArray.cols(); ++col) {
            eigenArray(row, col) = array[row][col].u.dbl;
        }
    }

    return eigenArray;
}

// helper function to create boundaryDescriptor from config file
std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> createBoundaryDescriptorFromConfig(
    json_value const& config, double const modelRadius) {
    // check for correct config
    if (config["height"].type == json_none) {
        return nullptr;
    }

    // extract descriptor coordinates from config, or create circular descriptor
    // if no coordinates are given
    if (config["coordinates"].type != json_none) {
        return std::make_shared<mpFlow::FEM::BoundaryDescriptor>(
            eigenFromJsonArray<double>(config["coordinates"]), config["height"].u.dbl);
    }
    else if ((config["width"].type != json_none) && (config["count"].type != json_none)) {
        return mpFlow::FEM::boundaryDescriptor::circularBoundary(
            config["count"].u.integer, config["width"].u.dbl, config["height"].u.dbl,
            modelRadius, config["offset"].u.dbl);
    }
    else {
        return nullptr;
    }
}

// helper to initialize mesh from config file
std::shared_ptr<mpFlow::numeric::IrregularMesh> createMeshFromConfig(
    json_value const& config, std::string const path,
    std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const boundaryDescriptor,
    cudaStream_t const cudaStream) {
    // check for correct config
    if (config["height"].type == json_none) {
        return nullptr;
    }

    // extract basic mesh parameter
    double height = config["height"];

    if (config["path"].type != json_none) {
        // load mesh from file
        std::string meshPath = str::format("%s/%s")(path, std::string(config["path"]));

        auto nodes = mpFlow::numeric::Matrix<double>::loadtxt(str::format("%s/nodes.txt")(meshPath), cudaStream);
        auto elements = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/elements.txt")(meshPath), cudaStream);
        auto boundary = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/boundary.txt")(meshPath), cudaStream);
        return std::make_shared<mpFlow::numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen(),
            boundary->toEigen(), height);
    }
    else if ((config["radius"].type != json_none) &&
            (config["outerEdgeLength"].type != json_none) &&
            (config["innerEdgeLength"].type != json_none)) {
        // fix mesh at boundaryDescriptor boundaries
        Eigen::ArrayXXd fixedPoints(boundaryDescriptor->count * 2, 2);
        for (unsigned i = 0; i < boundaryDescriptor->count; ++i) {
            fixedPoints.block(i * 2, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 0, 1, 2);
            fixedPoints.block(i * 2 + 1, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 2, 1, 2);
        }

        // create mesh with libdistmesh
        double const radius = config["radius"];
        auto distanceFuntion = distmesh::distanceFunction::circular(radius);
        auto dist_mesh = distmesh::distmesh(distanceFuntion, config["outerEdgeLength"],
            1.0 + (1.0 - (double)config["innerEdgeLength"] / (double)config["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::boundingBox(2), fixedPoints);

        // create mpflow matrix objects from distmesh arrays
        auto mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(std::get<0>(dist_mesh), std::get<1>(dist_mesh),
            distmesh::boundEdges(std::get<1>(dist_mesh)), height);

        // save mesh to files for later usage
        mkdir(str::format("%s/mesh")(path).c_str(), 0777);
        mpFlow::numeric::Matrix<double>::fromEigen(mesh->nodes, cudaStream)->savetxt(str::format("%s/mesh/nodes.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->elements, cudaStream)->savetxt(str::format("%s/mesh/elements.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->boundary, cudaStream)->savetxt(str::format("%s/mesh/boundary.txt")(path));

        return mesh;
    }
    else {
        return nullptr;
    }
}

// helper to create source descriptor from config file
template <class dataType>
std::shared_ptr<mpFlow::FEM::SourceDescriptor<dataType>> createSourceFromConfig(
    json_value const& config,
    std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const boundaryDescriptor,
    cudaStream_t const cudaStream) {
    // load excitation and measurement pattern from config or assume standard pattern, if not given
    std::shared_ptr<mpFlow::numeric::Matrix<int>> drivePattern = nullptr;
    if (config["drivePattern"].type != json_none) {
        drivePattern = matrixFromJsonArray<int>(config["drivePattern"], cudaStream);
    }
    else {
        drivePattern = mpFlow::numeric::Matrix<int>::eye(boundaryDescriptor->count, cudaStream);
    }

    std::shared_ptr<mpFlow::numeric::Matrix<int>> measurementPattern = nullptr;
    if (config["measurementPattern"].type != json_none) {
        measurementPattern = matrixFromJsonArray<int>(config["measurementPattern"], cudaStream);
    }
    else {
        measurementPattern = mpFlow::numeric::Matrix<int>::eye(boundaryDescriptor->count, cudaStream);
    }

    // read out currents
    std::vector<dataType> excitation(drivePattern->cols);
    if (config["value"].type == json_array) {
        for (unsigned i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = config["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dataType>(drivePattern->cols,
            config["value"].type != json_none ? config["value"].u.dbl : dataType(1));
    }

    // create source descriptor
    auto sourceType = std::string(config["type"]) == "voltage" ?
        mpFlow::FEM::SourceDescriptor<dataType>::Type::Fixed :
        mpFlow::FEM::SourceDescriptor<dataType>::Type::Open;
    auto source = std::make_shared<mpFlow::FEM::SourceDescriptor<dataType>>(sourceType,
        excitation, boundaryDescriptor, drivePattern, measurementPattern, cudaStream);

    return source;
}

template std::shared_ptr<mpFlow::FEM::SourceDescriptor<float>> createSourceFromConfig(
    json_value const&, std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<double>> createSourceFromConfig(
    json_value const&, std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<thrust::complex<float>>> createSourceFromConfig(
    json_value const&, std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<thrust::complex<double>>> createSourceFromConfig(
    json_value const&, std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const, cudaStream_t const);

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
