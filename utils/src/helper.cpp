#include <mpflow/mpflow.h>
#include <distmesh/distmesh.h>
#include <sys/stat.h>
#include "stringtools/format.hpp"
#include "json.h"
#include "helper.h"

// helper function to create an mpflow matrix from an json array
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> matrixFromJsonArray(
    json_value const& array, cudaStream_t const cudaStream) {
    // exctract sizes
    unsigned rows = array.u.array.length;
    unsigned cols = array[0].type == json_array ? array[0].u.array.length : 1;

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

// helper function to create boundaryDescriptor from config file
std::shared_ptr<mpFlow::FEM::BoundaryDescriptor> createBoundaryDescriptorFromConfig(
    json_value const& config, double const modelRadius) {
    // create boundaryDescriptor from config to fix mesh nodes to boundary nodes
    return mpFlow::FEM::boundaryDescriptor::circularBoundary(
        config["count"].u.integer, std::make_tuple(config["width"].u.dbl, config["height"].u.dbl),
        modelRadius, config["offset"].u.dbl);
}

// helper to initialize mesh from config file
std::shared_ptr<mpFlow::numeric::IrregularMesh> createMeshFromConfig(
    json_value const& config, std::string const path,
    std::shared_ptr<mpFlow::FEM::BoundaryDescriptor const> const boundaryDescriptor) {
    std::shared_ptr<mpFlow::numeric::IrregularMesh> mesh = nullptr;

    // extract basic mesh parameter
    double radius = config["radius"];
    double height = config["height"];

    if (config["meshPath"].type != json_none) {
        // load mesh from file
        std::string meshPath = str::format("%s/%s")(path, std::string(config["meshPath"]));

        auto nodes = mpFlow::numeric::Matrix<double>::loadtxt(str::format("%s/nodes.txt")(meshPath), nullptr);
        auto elements = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/elements.txt")(meshPath), nullptr);
        auto boundary = mpFlow::numeric::Matrix<int>::loadtxt(str::format("%s/boundary.txt")(meshPath), nullptr);
        mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen(),
            boundary->toEigen(), radius, height);
    }
    else {
        // fix mesh at boundaryDescriptor boundaries
        Eigen::ArrayXXd fixedPoints(boundaryDescriptor->count * 2, 2);
        for (unsigned i = 0; i < boundaryDescriptor->count; ++i) {
            fixedPoints.block(i * 2, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 0, 1, 2);
            fixedPoints.block(i * 2 + 1, 0, 1, 2) = boundaryDescriptor->coordinates.block(i, 2, 1, 2);
        }

        // create mesh with libdistmesh
        auto distanceFuntion = distmesh::distanceFunction::circular(radius);
        auto dist_mesh = distmesh::distmesh(distanceFuntion, config["outerEdgeLength"],
            1.0 + (1.0 - (double)config["innerEdgeLength"] / (double)config["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::boundingBox(2), fixedPoints);

        // create mpflow matrix objects from distmesh arrays
        mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(std::get<0>(dist_mesh), std::get<1>(dist_mesh),
            distmesh::boundEdges(std::get<1>(dist_mesh)), radius, height);

        // save mesh to files for later usage
        mkdir(str::format("%s/mesh")(path).c_str(), 0777);
        mpFlow::numeric::Matrix<double>::fromEigen(mesh->nodes, nullptr)->savetxt(str::format("%s/mesh/nodes.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->elements, nullptr)->savetxt(str::format("%s/mesh/elements.txt")(path));
        mpFlow::numeric::Matrix<int>::fromEigen(mesh->boundary, nullptr)->savetxt(str::format("%s/mesh/boundary.txt")(path));
    }

    return mesh;
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
        excitation = std::vector<dataType>(drivePattern->cols, config["value"].u.dbl);
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
