#include <mpflow/mpflow.h>
#include <distmesh/distmesh.h>
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

// helper to initialize mesh from config file
std::shared_ptr<mpFlow::numeric::IrregularMesh> createMeshFromConfig(
    json_value const& config, std::string const path) {
    std::shared_ptr<mpFlow::numeric::IrregularMesh> mesh = nullptr;

    // extract basic mesh parameter
    double radius = config["mesh"]["radius"];
    double height = config["mesh"]["height"];

    // create boundaryDescriptor from config to fix mesh nodes to boundary nodes
    auto boundaryDescriptor = mpFlow::FEM::boundaryDescriptor::circularBoundary(config["electrodes"]["count"].u.integer,
        std::make_tuple(config["electrodes"]["width"].u.dbl, config["electrodes"]["height"].u.dbl),
        radius, config["electrodes"]["offset"].u.dbl);

    if (config["mesh"]["meshPath"].type != json_none) {
        // load mesh from file
        std::string meshPath = str::format("%s/%s")(path, std::string(config["mesh"]["meshPath"]));

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
        auto distanceFuntion = distmesh::distance_function::circular(radius);
        auto dist_mesh = distmesh::distmesh(distanceFuntion, config["mesh"]["outerEdgeLength"],
            1.0 + (1.0 - (double)config["mesh"]["innerEdgeLength"] / (double)config["mesh"]["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::bounding_box(2), fixedPoints);

        // create mpflow matrix objects from distmesh arrays
        mesh = std::make_shared<mpFlow::numeric::IrregularMesh>(std::get<0>(dist_mesh), std::get<1>(dist_mesh),
            distmesh::boundedges(std::get<1>(dist_mesh)), radius, height);
    }

    return mesh;
}

// helper to create source descriptor from config file
template <class dataType>
std::shared_ptr<mpFlow::FEM::SourceDescriptor<dataType>> createSourceFromConfig(
    json_value const& config, cudaStream_t const cudaStream) {
    // create electrodes descriptor
    auto electrodes = mpFlow::FEM::boundaryDescriptor::circularBoundary(config["electrodes"]["count"].u.integer,
        std::make_tuple(config["electrodes"]["width"].u.dbl, config["electrodes"]["height"].u.dbl),
        config["mesh"]["radius"].u.dbl, config["electrodes"]["offset"].u.dbl);

    // load excitation and measurement pattern from config or assume standard pattern, if not given
    std::shared_ptr<mpFlow::numeric::Matrix<int>> drivePattern = nullptr;
    if (config["source"]["drivePattern"].type != json_none) {
        drivePattern = matrixFromJsonArray<int>(config["source"]["drivePattern"], cudaStream);
    }
    else {
        drivePattern = mpFlow::numeric::Matrix<int>::eye(electrodes->count, cudaStream);
    }

    std::shared_ptr<mpFlow::numeric::Matrix<int>> measurementPattern = nullptr;
    if (config["source"]["measurementPattern"].type != json_none) {
        measurementPattern = matrixFromJsonArray<int>(config["source"]["measurementPattern"], cudaStream);
    }
    else {
        measurementPattern = mpFlow::numeric::Matrix<int>::eye(electrodes->count, cudaStream);
    }

    // read out currents
    std::vector<dataType> excitation(drivePattern->cols);
    if (config["source"]["value"].type == json_array) {
        for (unsigned i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = config["source"]["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dataType>(drivePattern->cols, config["source"]["value"].u.dbl);
    }

    // create source descriptor
    auto sourceType = std::string(config["source"]["type"]) == "voltage" ?
        mpFlow::FEM::SourceDescriptor<dataType>::Type::Fixed :
        mpFlow::FEM::SourceDescriptor<dataType>::Type::Open;
    auto source = std::make_shared<mpFlow::FEM::SourceDescriptor<dataType>>(sourceType,
        excitation, electrodes, drivePattern, measurementPattern, cudaStream);

    return source;
}

template std::shared_ptr<mpFlow::FEM::SourceDescriptor<float>> createSourceFromConfig(
    json_value const&, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<double>> createSourceFromConfig(
    json_value const&, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<thrust::complex<float>>> createSourceFromConfig(
    json_value const&, cudaStream_t const);
template std::shared_ptr<mpFlow::FEM::SourceDescriptor<thrust::complex<double>>> createSourceFromConfig(
    json_value const&, cudaStream_t const);
