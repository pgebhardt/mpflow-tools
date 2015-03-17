#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "json.h"

using namespace mpFlow;

// use complex or real data type
#define dataType thrust::complex<double>

// helper function to create an mpflow matrix from an json array
template <class type>
std::shared_ptr<numeric::Matrix<type>> matrixFromJsonArray(const json_value& array, cudaStream_t cudaStream) {
    // exctract sizes
    unsigned rows = array.u.array.length;
    unsigned cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create matrix
    auto matrix = std::make_shared<numeric::Matrix<type>>(rows, cols, cudaStream);

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

int main(int argc, char* argv[]) {
    HighPrecisionTime time;

    // print out mpFlow version for refernce
    str::print("mpFlow version:", version::getVersionString());

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // load config document
    if (argc <= 1) {
        str::print("You need to give a path to the model config");
        return EXIT_FAILURE;
    }

    // extract filename and its path from command line arguments
    std::string filename = argv[1];
    auto filenamePos = filename.find_last_of("/");
    std::string path = filenamePos == std::string::npos ? "./" : filename.substr(0, filenamePos);

    str::print("----------------------------------------------------");
    str::print("Load model from config file:", argv[1]);

    std::ifstream file(filename);
    std::string fileContent((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    auto config = json_parse(fileContent.c_str(), fileContent.length());

    // check for success
    if (config == nullptr) {
        str::print("Error: Cannot parse config file");
        return EXIT_FAILURE;
    }

    // extract model config
    auto modelConfig = (*config)["model"];
    if (modelConfig.type == json_none) {
        str::print("Error: Invalid model config");
        return EXIT_FAILURE;
    }

    // Create Mesh using libdistmesh
    time.restart();
    str::print("----------------------------------------------------");

    // get mesh config
    auto meshConfig = modelConfig["mesh"];
    if (meshConfig.type == json_none) {
        str::print("Error: Invalid model config");
        return EXIT_FAILURE;
    }

    // create mesh from config or load from files, if mesh dir is given
    std::shared_ptr<numeric::IrregularMesh> mesh = nullptr;
    double radius = meshConfig["radius"];

    // create electrodes descriptor
    double offset = modelConfig["electrodes"]["offset"];
    auto electrodes = FEM::boundaryDescriptor::circularBoundary(modelConfig["electrodes"]["count"].u.integer,
        std::make_tuple(modelConfig["electrodes"]["width"].u.dbl, modelConfig["electrodes"]["height"].u.dbl),
        radius, offset);

    if (meshConfig["meshPath"].type != json_none) {
        // load mesh from file
        std::string meshPath = str::format("%s/%s")(path, std::string(meshConfig["meshPath"]));
        str::print("Load mesh from files:", meshPath);

        auto nodes = numeric::Matrix<double>::loadtxt(str::format("%s/nodes.txt")(meshPath), cudaStream);
        auto elements = numeric::Matrix<int>::loadtxt(str::format("%s/elements.txt")(meshPath), cudaStream);
        auto boundary = numeric::Matrix<int>::loadtxt(str::format("%s/boundary.txt")(meshPath), cudaStream);
        mesh = std::make_shared<numeric::IrregularMesh>(nodes->toEigen(), elements->toEigen(),
            boundary->toEigen(), radius, (double)meshConfig["height"]);

        str::print("Mesh loaded with", nodes->rows, "nodes and", elements->rows, "elements");
        str::print("Time:", time.elapsed() * 1e3, "ms");
    }
    else {
        str::print("Create mesh using libdistmesh");

        // fix mesh at electrodes boundaries
        Eigen::ArrayXXd fixedPoints(electrodes->count * 2, 2);
        for (unsigned electrode = 0; electrode < electrodes->count; ++electrode) {
            fixedPoints.block(electrode * 2, 0, 1, 2) = electrodes->coordinates.block(electrode, 0, 1, 2);
            fixedPoints.block(electrode * 2 + 1, 0, 1, 2) = electrodes->coordinates.block(electrode, 2, 1, 2);
        }

        // create mesh with libdistmesh
        auto distanceFuntion = distmesh::distance_function::circular(radius);
        auto dist_mesh = distmesh::distmesh(distanceFuntion, meshConfig["outerEdgeLength"],
            1.0 + (1.0 - (double)meshConfig["innerEdgeLength"] / (double)meshConfig["outerEdgeLength"]) *
            distanceFuntion / radius, 1.1 * radius * distmesh::bounding_box(2), fixedPoints);

        str::print("Mesh created with", std::get<0>(dist_mesh).rows(), "nodes and",
            std::get<1>(dist_mesh).rows(), "elements");
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // create mpflow matrix objects from distmesh arrays
        mesh = std::make_shared<numeric::IrregularMesh>(std::get<0>(dist_mesh), std::get<1>(dist_mesh),
            distmesh::boundedges(std::get<1>(dist_mesh)), radius, (double)meshConfig["height"]);
    }

    // Create main model class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main model class");

    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, false>>(
        mesh, electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Create model helper classes
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create model helper classes");

    // load excitation and measurement pattern from config or assume standard pattern, if not given
    std::shared_ptr<numeric::Matrix<int>> drivePattern = nullptr;
    if (modelConfig["source"]["drivePattern"].type != json_none) {
        drivePattern = matrixFromJsonArray<int>(modelConfig["source"]["drivePattern"], cudaStream);
    }
    else {
        drivePattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);;
    }

    std::shared_ptr<numeric::Matrix<int>> measurementPattern = nullptr;
    if (modelConfig["source"]["measurementPattern"].type != json_none) {
        measurementPattern = matrixFromJsonArray<int>(modelConfig["source"]["measurementPattern"], cudaStream);
    }
    else {
        measurementPattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);;
    }

    // read out currents
    std::vector<dataType> excitation(drivePattern->cols);
    if (modelConfig["source"]["value"].type == json_array) {
        for (unsigned i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = modelConfig["source"]["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dataType>(drivePattern->cols, (double)modelConfig["source"]["value"].u.dbl);
    }

    // create source descriptor
    auto sourceType = std::string(modelConfig["source"]["type"]) == "voltage" ?
        FEM::SourceDescriptor<dataType>::Type::Fixed : FEM::SourceDescriptor<dataType>::Type::Open;
    auto source = std::make_shared<FEM::SourceDescriptor<dataType>>(sourceType, excitation, electrodes,
        drivePattern, measurementPattern, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // load predefined gamma distribution from file, if path is given
    std::shared_ptr<numeric::Matrix<dataType>> gamma = nullptr;
    if (modelConfig["gammaFile"].type != json_none) {
        gamma = numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(modelConfig["gammaFile"])), cudaStream);
    }
    else {
        gamma = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), 1, cudaStream, 1.0f);
    }

    // Create forward solver and solve potential
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Solve electrical potential for all excitations");

    // use different numeric solver for different source types
    std::shared_ptr<numeric::Matrix<dataType> const> result = nullptr, potential = nullptr;
    unsigned steps = 0;
    if (sourceType == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<numeric::BiCGSTAB, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, 1e-15, &steps);
        potential = forwardSolver->phi[0];
    }
    else {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<numeric::ConjugateGradient, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, 1e-15, &steps);
        potential = forwardSolver->phi[0];
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps);

    // Print result and save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    str::print("----------------------------------------------------");
    str::print("Result:");
    str::print(*result);
    result->savetxt(str::format("%s/result.txt")(path));

    // save potential to file
    potential->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    potential->savetxt(str::format("%s/potential.txt")(path));

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
