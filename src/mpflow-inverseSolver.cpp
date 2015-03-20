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
    if (argc <= 3) {
        str::print("You need to give filenames of refernce and measurement data");
        return EXIT_FAILURE;
    }

    auto refernce = numeric::Matrix<dataType>::loadtxt(argv[2], cudaStream);
    auto measurement = numeric::Matrix<dataType>::loadtxt(argv[3], cudaStream);

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

    auto mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes);

    str::print("Mesh loaded with", mesh->nodes.rows(), "nodes and", mesh->elements.rows(), "elements");
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Create main model class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main model class");

    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, false>>(
        mesh, source->electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // load predefined gamma distribution from file, if path is given
    std::shared_ptr<numeric::Matrix<dataType>> gamma = nullptr;
    if (modelConfig["gammaFile"].type != json_none) {
        gamma = numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(modelConfig["gammaFile"])), cudaStream);
    }
    else {
        gamma = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), 1, cudaStream, 1.0);
    }

    // Create solver and reconstruct image
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create solver and reconstruct image");

    // extract model config
    auto solverConfig = (*config)["solver"];
    if (solverConfig.type == json_none) {
        str::print("Error: Invalid solver config");
        return EXIT_FAILURE;
    }

    // use different numeric solver for different source types
    std::shared_ptr<numeric::Matrix<dataType> const> result = nullptr;
    if (source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        auto solver = std::make_shared<EIT::Solver<numeric::BiCGSTAB, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer,
            1, modelConfig["regularizationFactor"].u.dbl, cublasHandle, cudaStream);

        // set loaded gamma distribution
        solver->gamma->copy(gamma, cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        // copy refernce and measurement to solver
        solver->calculation[0]->copy(refernce, cudaStream);
        solver->measurement[0]->copy(measurement, cudaStream);

        cudaStreamSynchronize(cudaStream);
        time.restart();

        result = solver->solveDifferential(cublasHandle, cudaStream);
    }
    else {
        auto solver = std::make_shared<EIT::Solver<numeric::ConjugateGradient, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer,
            1, modelConfig["regularizationFactor"].u.dbl, cublasHandle, cudaStream);

        // set loaded gamma distribution
        solver->gamma->copy(gamma, cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        // copy refernce and measurement to solver
        solver->calculation[0]->copy(refernce, cudaStream);
        solver->measurement[0]->copy(measurement, cudaStream);

        cudaStreamSynchronize(cudaStream);
        time.restart();

        result = solver->solveDifferential(cublasHandle, cudaStream);
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    result->savetxt(str::format("%s/reconstruction.txt")(path));

    // Save mesh

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
