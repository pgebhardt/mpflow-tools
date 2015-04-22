#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "json.h"
#include "helper.h"

using namespace mpFlow;

// use complex or real data type
#define dataType thrust::complex<double>

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

    // load reference and measurement data
    if (argc <= 2) {
        str::print("You need to give filenames of refernce and/or measurement data");
        return EXIT_FAILURE;
    }

    std::shared_ptr<numeric::Matrix<dataType>> reference = nullptr, measurement = nullptr;
    if (argc == 3) {
        measurement = numeric::Matrix<dataType>::loadtxt(argv[2], cudaStream);
        reference = std::make_shared<numeric::Matrix<dataType>>(measurement->rows,
            measurement->cols, cudaStream);
    }
    else {
        reference = numeric::Matrix<dataType>::loadtxt(argv[2], cudaStream);
        measurement = numeric::Matrix<dataType>::loadtxt(argv[3], cudaStream);
    }

    // Create model helper classes
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create model helper classes");

    auto electrodes = createBoundaryDescriptorFromConfig(modelConfig["electrodes"], modelConfig["mesh"]["radius"].u.dbl);
    auto source = createSourceFromConfig<dataType>(modelConfig["source"], electrodes, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // load mesh from config
    time.restart();
    str::print("----------------------------------------------------");

    auto mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes, cudaStream);

    str::print("Mesh loaded with", mesh->nodes.rows(), "nodes and", mesh->elements.rows(), "elements");
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Create main model class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main model class");

    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, false>>(
        mesh, source->electrodes, 1.0, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create solver and reconstruct image");

    // extract model config
    auto solverConfig = (*config)["solver"];
    if (solverConfig.type == json_none) {
        str::print("Error: Invalid solver config");
        return EXIT_FAILURE;
    }

    // extract parallel images count
    int const parallelImages = std::max(1, (int)solverConfig["parallelImages"].u.integer);

    // Create solver and reconstruct image
    // use different numeric solver for different source types
    std::shared_ptr<numeric::Matrix<dataType> const> result = nullptr;
    unsigned steps = 0;
    if (source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        auto solver = std::make_shared<EIT::Solver<numeric::BiCGSTAB, decltype(equation)::element_type>>(
            equation, source, std::max(1, (int)modelConfig["componentsCount"].u.integer),
            parallelImages, solverConfig["regularizationFactor"].u.dbl, cublasHandle, cudaStream);

        // set material distribution and initialize models
        solver->gamma->fill(dataType(modelConfig["material"].u.dbl), cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        // copy refernce and measurement to solver
        for (auto cal : solver->calculation) {
            cal->copy(reference, cudaStream);
        }
        for (auto mes : solver->measurement) {
            mes->copy(measurement, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        time.restart();

        result = solver->solveDifferential(cublasHandle, cudaStream, 0, &steps);
    }
    else {
        auto solver = std::make_shared<EIT::Solver<numeric::ConjugateGradient, decltype(equation)::element_type>>(
            equation, source, std::max(1, (int)modelConfig["componentsCount"].u.integer),
            parallelImages, solverConfig["regularizationFactor"].u.dbl, cublasHandle, cudaStream);

        // set material distribution and initialize models
        solver->gamma->fill(dataType(modelConfig["material"].u.dbl), cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        // copy refernce and measurement to solver
        for (auto cal : solver->calculation) {
            cal->copy(reference, cudaStream);
        }
        for (auto mes : solver->measurement) {
            mes->copy(measurement, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        time.restart();

        result = solver->solveDifferential(cublasHandle, cudaStream, 0, &steps);
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time per image:", time.elapsed() * 1e3 / parallelImages, "ms, FPS:",
        parallelImages / time.elapsed(), "Hz, Iterations:", steps);

    // Save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    result->savetxt(str::format("%s/reconstruction.txt")(path));

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
