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
    template <class> class numericalSolverType,
    bool logarithmic
>
void solveInverseModelFromConfig(int argc, char* argv[], json_value const& config,
    cublasHandle_t const cublasHandle, cudaStream_t const cudaStream) {
    HighPrecisionTime time;

    // extract path from command line arguments
    std::string const filename = argv[1];
    auto const filenamePos = filename.find_last_of("/");
    std::string const path = filenamePos == std::string::npos ? "./" : filename.substr(0, filenamePos);

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

    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create and initialize solver");

    auto solver = solver::Solver<EIT::ForwardSolver<numericalSolverType,
        FEM::Equation<dataType, FEM::basis::Linear, logarithmic>>,
        numericalSolverType>::fromConfig(config, cublasHandle, cudaStream, path);

    // copy measurement and reference data to solver
    for (auto mes : solver->measurement) {
        mes->copy(measurement, cudaStream);
    }
    for (auto cal : solver->calculation) {
        cal->copy(reference, cudaStream);
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // read out amount of newton and numerical solver steps for reconstruction
    int const maxIterations = config["solver"]["maxIterations"].u.integer;
    int const newtonSteps = std::max(1, (int)config["solver"]["steps"].u.integer);

    str::print("----------------------------------------------------");
    str::print("Reconstruct image");

    // reconstruct image
    std::shared_ptr<numeric::Matrix<dataType> const> result = nullptr;
    if (newtonSteps == 1) {
        time.restart();

        unsigned steps = 0;
        result = solver->solveDifferential(cublasHandle, cudaStream, maxIterations, &steps);
    
        cudaStreamSynchronize(cudaStream);
        str::print("Time per image:", time.elapsed() * 1e3 / solver->measurement.size(), "ms, FPS:",
            solver->measurement.size() / time.elapsed(), "Hz, Iterations:", steps);        
    }
    else {
        // correct measurement
        solver->calculation[0]->scalarMultiply(-1.0, cudaStream);
        solver->measurement[0]->add(solver->calculation[0], cudaStream);
        solver->measurement[0]->add(solver->forwardModel->result, cudaStream);    
        
        cudaStreamSynchronize(cudaStream);
        time.restart();

        result = solver->solveAbsolute(newtonSteps, cublasHandle, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");                        
        
    }

    // Save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    
    if (logarithmic) {
        numeric::Matrix<dataType>::fromEigen(
            solver->forwardModel->equation->referenceValue * (log(10.0) * result->toEigen() / 10.0).exp(),
            cudaStream)->savetxt(str::format("%s/reconstruction.txt")(path));        
    }
    else {
        result->savetxt(str::format("%s/reconstruction.txt")(path));
    }
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
    bool constexpr logarithmic = false;
    if (std::string(modelConfig["source"]["type"]) == "current") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveInverseModelFromConfig<double, numeric::ConjugateGradient, logarithmic>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveInverseModelFromConfig<thrust::complex<double>, numeric::ConjugateGradient, logarithmic>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
    }
    else if (std::string(modelConfig["source"]["type"]) == "voltage") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveInverseModelFromConfig<double, numeric::BiCGSTAB, logarithmic>(
                argc, argv, *config, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveInverseModelFromConfig<thrust::complex<double>, numeric::BiCGSTAB, logarithmic>(
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
