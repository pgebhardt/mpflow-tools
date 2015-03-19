#ifndef MPFLOW_TOOLS_UTILS_HELPER_H
#define MPFLOW_TOOLS_UTILS_HELPER_H

// helper function to create an mpflow matrix from an json array
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> matrixFromJsonArray(
    json_value const& array, cudaStream_t const cudaStream);

// helper to initialize mesh from config file
std::shared_ptr<mpFlow::numeric::IrregularMesh> createMeshFromConfig(
    json_value const& config, std::string const path);

// helper to create source descriptor from config file
template <class dataType>
std::shared_ptr<mpFlow::FEM::SourceDescriptor<dataType>> createSourceFromConfig(
    json_value const& config, cudaStream_t const cudaStream);

#endif
