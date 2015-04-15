#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "helper.h"

using namespace mpFlow;
#define RADIUS (0.085)

int main(int argc, char* argv[]) {
    HighPrecisionTime time;

    // check command line arguments
    if (argc <= 1) {
        str::print("You need to give a path to the model config");
        return EXIT_FAILURE;
    }

    // extrac maximum pipeline length
    int const maxPipelineLenght = argc > 2 ? atoi(argv[2]) : 512;

    // print out basic system info for reference
    str::print("----------------------------------------------------");
    str::print("mpFlow", version::getVersionString(),
        str::format("(%s %s)")(__DATE__, __TIME__));
    str::print(str::format("[%s] on %s")(getCompilerName(), _TARGET_ARCH_NAME_));
    str::print("----------------------------------------------------");
    printCudaDeviceProperties();
    str::print("----------------------------------------------------");
    str::print("Config file:", argv[1]);

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;

    cudaSetDevice(argc <= 3 ? 0 : cudaSetDevice(atoi(argv[3])));
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // extract filename and its path from command line arguments
    std::string const filename = argv[1];
    std::string const path = filename.find_last_of("/") == std::string::npos ?
        "./" : filename.substr(0, filename.find_last_of("/"));

    // load config from file
    std::ifstream file(filename);
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

    // Create model helper classes
    auto const electrodes = createBoundaryDescriptorFromConfig(modelConfig["electrodes"], modelConfig["mesh"]["radius"].u.dbl);
    auto const source = createSourceFromConfig<float>(modelConfig["source"], electrodes, cudaStream);

    time.restart();
    str::print("----------------------------------------------------");

    // load mesh from config
    auto const mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes);

    str::print("Mesh loaded with", mesh->nodes.rows(), "nodes and", mesh->elements.rows(), "elements");
    str::print("Time:", time.elapsed() * 1e3, "ms");

    str::print("----------------------------------------------------");
    str::print("Initialize forward model");
    time.restart();

    // Create main model class
    auto equation = std::make_shared<FEM::Equation<float, FEM::basis::Linear, false>>(
        mesh, source->electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // Create solver class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main solver class");

    auto solver = std::make_shared<EIT::Solver<numeric::ConjugateGradient,
        typename decltype(equation)::element_type>>(
        equation, source, 7, 1, 0.0, cublasHandle, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // initialize solver
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Solve forward model and initialize inverse solver matrices");

    solver->preSolve(cublasHandle, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    str::print("----------------------------------------------------");
    str::print("Reconstruct images for different pipeline lengths:\n");

    Eigen::ArrayXd result = Eigen::ArrayXd::Zero(maxPipelineLenght);
    for (unsigned length = 1; length <= maxPipelineLenght; ++length) {
        // create inverse solver
        solver = std::make_shared<EIT::Solver<numeric::ConjugateGradient,
            typename decltype(equation)::element_type>>(
            equation, source, 7, length, 1e-4, cublasHandle, cudaStream);
        solver->preSolve(cublasHandle, cudaStream);

        // clear reference scenario to force solve to calculate all iteration steps
        for (auto ref : solver->calculation) {
            ref->scalarMultiply(0.0, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        time.restart();

        for (unsigned i = 0; i < 10; ++i) {
            solver->solveDifferential(cublasHandle, cudaStream,
                solver->gamma->rows / 8);
        }

        cudaStreamSynchronize(cudaStream);
        result(length - 1) = time.elapsed() / 10.0;

        if (length % 16 == 0) {
            str::print("pipeline length:", length, "; time:",
                result(length - 1) * 1e3, "ms ; fps:",
                (double)length / result(length - 1));
        }
    }

    // save benchmark result
    numeric::Matrix<double>::fromEigen(result, cudaStream)->savetxt(
        str::format("%s/inverseBenchmarkResult.txt")(path));

    return EXIT_SUCCESS;
}
