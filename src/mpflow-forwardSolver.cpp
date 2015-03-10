#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "json.h"

using namespace mpFlow;

// define tolerance for single or double precision
#ifdef USE_DOUBLE
    #define tolerance (1e-15)
#else
    #define tolerance (1e-9)
#endif

// helper function to create an mpflow matrix from an json array
template <class type>
std::shared_ptr<numeric::Matrix<type>> matrixFromJsonArray(const json_value& array, cudaStream_t cudaStream) {
    // exctract sizes
    dtype::size rows = array.u.array.length;
    dtype::size cols = array[0].type == json_array ? array[0].u.array.length : 1;

    // create matrix
    auto matrix = std::make_shared<numeric::Matrix<type>>(rows, cols, cudaStream);

    // exctract values
    if (array[0].type != json_array) {
        for (dtype::index row = 0; row < matrix->rows; ++row) {
            (*matrix)(row, 0) = array[row].u.dbl;
        }
    }
    else {
        for (dtype::index row = 0; row < matrix->rows; ++row)
        for (dtype::index col = 0; col < matrix->cols; ++col) {
            (*matrix)(row, col) = array[row][col].u.dbl;
        }
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
}

void setCircularRegion(float x, float y, float radius,
    float value, std::shared_ptr<numeric::IrregularMesh> mesh,
    std::shared_ptr<numeric::Matrix<dtype::real>> gamma) {
    for (dtype::index element = 0; element < mesh->elements->rows; ++element) {
        auto nodes = mesh->elementNodes(element);

        dtype::real midX = 0.0, midY = 0.0;
        for (dtype::index node = 0; node < nodes.size(); ++node) {
            midX += std::get<0>(std::get<1>(nodes[node])) / nodes.size();
            midY += std::get<1>(std::get<1>(nodes[node])) / nodes.size();
        }
        if (math::square(midX - x) + math::square(midY - y) <= math::square(radius)) {
            (*gamma)(element, 0) = value;
        }
    }
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

    str::print("----------------------------------------------------");
    str::print("Load model from config file:", argv[1]);

    std::ifstream file(argv[1]);
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
        std::string meshDir(meshConfig["meshPath"]);
        str::print("Load mesh from files:", meshDir);

        auto nodes = numeric::Matrix<dtype::real>::loadtxt(str::format("%s/nodes.txt")(meshDir), cudaStream);
        auto elements = numeric::Matrix<dtype::index>::loadtxt(str::format("%s/elements.txt")(meshDir), cudaStream);
        auto boundary = numeric::Matrix<dtype::index>::loadtxt(str::format("%s/boundary.txt")(meshDir), cudaStream);
        mesh = std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, (double)meshConfig["height"]);

        str::print("Mesh loaded with", nodes->rows, "nodes and", elements->rows, "elements");
        str::print("Time:", time.elapsed() * 1e3, "ms");
    }
    else {
        str::print("Create mesh using libdistmesh");

        // fix mesh at electrodes boundaries
        Eigen::ArrayXXd fixedPoints(electrodes->count * 2, 2);
        for (dtype::index electrode = 0; electrode < electrodes->count; ++electrode) {
            fixedPoints(electrode * 2, 0) = std::get<0>(std::get<0>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2, 1) = std::get<1>(std::get<0>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2 + 1, 0) = std::get<0>(std::get<1>(electrodes->coordinates[electrode]));
            fixedPoints(electrode * 2 + 1, 1) = std::get<1>(std::get<1>(electrodes->coordinates[electrode]));
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
        auto nodes = numeric::matrix::fromEigen<dtype::real, double>(std::get<0>(dist_mesh));
        auto elements = numeric::matrix::fromEigen<dtype::index, int>(std::get<1>(dist_mesh));
        auto boundary = numeric::matrix::fromEigen<dtype::index, int>(distmesh::boundedges(std::get<1>(dist_mesh)));
        mesh = std::make_shared<numeric::IrregularMesh>(nodes, elements, boundary, radius, (double)meshConfig["height"]);
    }

    // Create model helper classes
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create model helper classes");

    // load excitation and measurement pattern from config or assume standard pattern, if not given
    std::shared_ptr<numeric::Matrix<dtype::real>> drivePattern = nullptr;
    if (modelConfig["source"]["drivePattern"].type != json_none) {
        drivePattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["drivePattern"], cudaStream);
    }
    else {
        drivePattern = numeric::Matrix<dtype::real>::eye(electrodes->count, cudaStream);;
    }

    std::shared_ptr<numeric::Matrix<dtype::real>> measurementPattern = nullptr;
    if (modelConfig["source"]["measurementPattern"].type != json_none) {
        measurementPattern = matrixFromJsonArray<dtype::real>(modelConfig["source"]["measurementPattern"], cudaStream);
    }
    else {
        measurementPattern = numeric::Matrix<dtype::real>::eye(electrodes->count, cudaStream);;
    }

    // read out currents
    std::vector<dtype::real> excitation(drivePattern->cols);
    if (modelConfig["source"]["value"].type == json_array) {
        for (dtype::index i = 0; i < drivePattern->cols; ++i) {
            excitation[i] = modelConfig["source"]["value"][i].u.dbl;
        }
    }
    else {
        excitation = std::vector<dtype::real>(drivePattern->cols, (dtype::real)modelConfig["source"]["value"].u.dbl);
    }

    // create source descriptor
    auto sourceType = std::string(modelConfig["source"]["type"]) == "voltage" ?
        FEM::SourceDescriptor::Type::Fixed : FEM::SourceDescriptor::Type::Open;
    auto source = std::make_shared<FEM::SourceDescriptor>(sourceType, excitation, electrodes,
        drivePattern, measurementPattern, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Create main model class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main model class");

    auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
        mesh, electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // load predefined gamma distribution from file, if path is given
    std::shared_ptr<numeric::Matrix<dtype::real>> gamma = nullptr;
    if (modelConfig["gammaFile"].type != json_none) {
        gamma = numeric::Matrix<dtype::real>::loadtxt(std::string(modelConfig["gammaFile"]), cudaStream);
    }
    else {
        gamma = std::make_shared<numeric::Matrix<dtype::real>>(mesh->elements->rows, 1, cudaStream);
    }

    // Create forward solver and solve potential
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Solve electrical potential for all excitations");

    // use different numeric solver for different source types
    std::shared_ptr<numeric::Matrix<dtype::real>> result = nullptr, potential = nullptr;
    dtype::index steps = 0;
    if (sourceType == FEM::SourceDescriptor::Type::Fixed) {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear, numeric::BiCGSTAB>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, tolerance, &steps);
        potential = forwardSolver->phi[0];
    }
    else {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear, numeric::ConjugateGradient>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        time.restart();
        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, tolerance, &steps);
        potential = forwardSolver->phi[0];
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps, "Tolerance:", tolerance);

    // Print result
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    str::print("----------------------------------------------------");
    str::print("Result:");
    result->savetxt(&std::cout);
    result->savetxt("result.txt");

    // save potential to file
    potential->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    potential->savetxt("phi.txt");

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
