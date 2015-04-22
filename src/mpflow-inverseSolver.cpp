#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "json.h"
#include "helper.h"

using namespace mpFlow;

template <
    class dataType,
    template <class> class numericalSolverType
>
void solveInverseModelFromConfig(int argc, char* argv[], json_value const& config,
    cublasHandle_t const cublasHandle, cudaStream_t const cudaStream) {
    HighPrecisionTime time;

    // extract path from command line arguments
    std::string const filename = argv[1];
    auto const filenamePos = filename.find_last_of("/");
    std::string const path = filenamePos == std::string::npos ? "./" : filename.substr(0, filenamePos);

    // extract model config
    auto const modelConfig = config["model"];
    if (modelConfig.type == json_none) {
        str::print("Error: Invalid model config");
        return;
    }

    // extract model config
    auto const solverConfig = config["solver"];
    if (solverConfig.type == json_none) {
        str::print("Error: Invalid solver config");
        return;
    }

    // load measurement and reference data
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

    auto const electrodes = createBoundaryDescriptorFromConfig(modelConfig["electrodes"],
        modelConfig["mesh"]["radius"].u.dbl);
    auto const source = createSourceFromConfig<dataType>(modelConfig["source"],
        electrodes, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // load mesh from config
    time.restart();
    str::print("----------------------------------------------------");

    auto const mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes, cudaStream);

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

    // extract parallel images count
    int const parallelImages = std::max(1, (int)solverConfig["parallelImages"].u.integer);

    // create inverse solver and forward model
    auto solver = std::make_shared<EIT::Solver<numericalSolverType,
        typename decltype(equation)::element_type>>(
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

    // reconstruct image(s)
    unsigned steps = 0;
    auto const result = solver->solveDifferential(cublasHandle, cudaStream, 0, &steps);

    cudaStreamSynchronize(cudaStream);
    str::print("Time per image:", time.elapsed() * 1e3 / parallelImages, "ms, FPS:",
        parallelImages / time.elapsed(), "Hz, Iterations:", steps);

    // Save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    result->savetxt(str::format("%s/reconstruction.txt")(path));
}

int main(int argc, char* argv[]) {
    // check command line arguments
    if (argc <= 1) {
        str::print("You need to give a path to the model config");
        return EXIT_FAILURE;
    }
    if (argc <= 2) {
        str::print("You need to give filenames of reference and/or measurement data");
        return EXIT_FAILURE;
    }

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // print out basic system info for reference
    str::print("----------------------------------------------------");
    str::print("mpFlow", version::getVersionString(),
        str::format("(%s %s)")(__DATE__, __TIME__));
    str::print(str::format("[%s] on %s")(getCompilerName(), _TARGET_ARCH_NAME_));
    str::print("----------------------------------------------------");
    printCudaDeviceProperties();
    str::print("----------------------------------------------------");
    str::print("Config file:", argv[1]);

    str::print("----------------------------------------------------");
    str::print("Load model from config file:", argv[1]);

    std::ifstream file(argv[1]);
    std::string const fileContent((std::istreambuf_iterator<char>(file)),
        std::istreambuf_iterator<char>());
    auto const config = json_parse(fileContent.c_str(), fileContent.length());

    // check for success
    if (config == nullptr) {
        str::print("Error: Cannot parse config file");
        return EXIT_FAILURE;
    }

    // extract model config
    auto const modelConfig = (*config)["model"];
    if (modelConfig.type == json_none) {
        str::print("Error: Invalid model config");
        return EXIT_FAILURE;
    }

    // extract data type from model config and solve inverse problem
    // use different numerical solver for different source types
    if (std::string(modelConfig["source"]["type"]) == "current") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveInverseModelFromConfig<double, numeric::ConjugateGradient>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveInverseModelFromConfig<thrust::complex<double>, numeric::ConjugateGradient>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
    }
    else if (std::string(modelConfig["source"]["type"]) == "voltage") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveInverseModelFromConfig<double, numeric::BiCGSTAB>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveInverseModelFromConfig<thrust::complex<double>, numeric::BiCGSTAB>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
    }
    else {
        str::print("Error: Invalid model config");
        return EXIT_FAILURE;
    }

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
