#ifndef MPFLOW_TOOLS_UTILS_HELPER_H
#define MPFLOW_TOOLS_UTILS_HELPER_H

#include "json.h"

// helper function to create an mpflow matrix from an json array
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> matrixFromJsonArray(
    json_value const& array, cudaStream_t const cudaStream);

// creates eigen array from an json array
template <class type>
Eigen::Array<type, Eigen::Dynamic, Eigen::Dynamic> eigenFromJsonArray(
    json_value const& array);

// helper function to create boundaryDescriptor from config file
std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> createBoundaryDescriptorFromConfig(
    json_value const& config, double const modelRadius);

// helper to initialize mesh from config file
std::shared_ptr<mpFlow::numeric::IrregularMesh> createMeshFromConfig(
    json_value const& config, std::string const path,
    std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const boundaryDescriptor);

// helper to create source descriptor from config file
template <class dataType>
std::shared_ptr<mpFlow::FEM::SourceDescriptor<dataType>> createSourceFromConfig(
    json_value const& config,
    std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const boundaryDescriptor,
    cudaStream_t const cudaStream);

// print properties of current cuda device
void printCudaDeviceProperties();

// get basic compiler name
std::string getCompilerName();

#endif
