#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "helper.h"

using namespace mpFlow;

// plot result for complex types with real and imaginary part
template <typename dataType>
void printResult(std::shared_ptr<numeric::Matrix<dataType> const> const result) {
    str::print(result->toEigen());
}

template <typename dataType>
void printResult(std::shared_ptr<numeric::Matrix<thrust::complex<dataType>> const> const result) {
    str::print("Real Part:");
    str::print(result->toEigen().real());
    str::print("Imaginary Part:");
    str::print(result->toEigen().imag());
}

template <
    class dataType,
    template <class> class numericalSolverType
>
void solveForwardModelFromConfig(json_value const& config, std::string const path,
    cublasHandle_t const cublasHandle, cudaStream_t const cudaStream) {
    HighPrecisionTime time;

    str::print("----------------------------------------------------");
    str::print("Load forward solver");
    time.restart();

    // Create forward solver and solve potential
    auto forwardSolver = [=]() {
        typedef models::EIT<numericalSolverType, FEM::Equation<dataType, FEM::basis::Linear, false>> modelType;
            
        try {
            return modelType::fromConfig(config, cublasHandle, cudaStream, path);
        }
        catch (std::exception const& exception) {
            str::print("Error: Invalid config file");
            str::print("Error message:", exception.what());
            
            return std::shared_ptr<modelType>(nullptr);            
        }
    }();
    if (forwardSolver == nullptr) {
        return;
    }

    // load predefined material distribution from file, if path is given
    auto const material = [=](json_value const& material) -> std::shared_ptr<numeric::Matrix<dataType>> {
        if (material.type == json_string) {
            return numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(material)), cudaStream);
        }
        else if (material.type == json_double) {
            return std::make_shared<numeric::Matrix<dataType>>(forwardSolver->mesh->elements.rows(), 1,
                cudaStream, dataType(material.u.dbl));
        }
        else {
            return nullptr;
        }
    }(config["material"]);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    str::print("----------------------------------------------------");
    str::print("Solve electrical potential for all excitations");
    time.restart();

    // solve forward model
    unsigned steps = 0;
    auto result = forwardSolver->solve(material, cublasHandle, cudaStream, &steps);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");
    str::print("Steps:", steps);

    // calculate electrical potential at cross section from 2.5D model
    auto potential = std::make_shared<numeric::Matrix<dataType>>(forwardSolver->phi[0]->rows,
        forwardSolver->phi[0]->cols, cudaStream);
    for (auto const phi : forwardSolver->phi) {
        potential->add(phi, cudaStream);
    }

    // Print and save results
    result->copyToHost(cudaStream);
    potential->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    str::print("----------------------------------------------------");
    str::print("Result:");
    printResult(result);
    result->savetxt(str::format("%s/result.txt")(path));
    potential->savetxt(str::format("%s/potential.txt")(path));
}

int main(int argc, char* argv[]) {
    // check command line arguments
    if (argc <= 1) {
        str::print("You need to give a path to the model config");
        return EXIT_FAILURE;
    }

    // init cuda
    cudaSetDevice(argc <= 2 ? 0 : cudaSetDevice(atoi(argv[2])));
    cudaStream_t const cudaStream = []{ cudaStream_t stream; cudaStreamCreate(&stream); return stream; }();
    cublasHandle_t const cublasHandle = []{ cublasHandle_t handle; cublasCreate(&handle); return handle; }();

    // print out basic system info for reference
    str::print("----------------------------------------------------");
    str::print("mpFlow", version::getVersionString(),
        str::format("(%s %s)")(__DATE__, __TIME__));
    str::print(str::format("[%s] on %s")(getCompilerName(), _TARGET_ARCH_NAME_));
    str::print("----------------------------------------------------");
    printCudaDeviceProperties();
    str::print("----------------------------------------------------");
    str::print("Config file:", argv[1]);

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

    // extract data type from model config and solve forward model
    // use different numerical solver for different source types
    if (std::string(modelConfig["source"]["type"]) == "current") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveForwardModelFromConfig<double, numeric::ConjugateGradient>(
                modelConfig, path, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveForwardModelFromConfig<thrust::complex<double>, numeric::ConjugateGradient>(
                modelConfig, path, cublasHandle, cudaStream);
        }
    }
    else if (std::string(modelConfig["source"]["type"]) == "voltage") {
        if ((modelConfig["numericType"].type == json_none) ||
            (std::string(modelConfig["numericType"]) == "real")) {
            solveForwardModelFromConfig<double, numeric::BiCGSTAB>(
                modelConfig, path, cublasHandle, cudaStream);
        }
        else if (std::string(modelConfig["numericType"]) == "complex") {
            solveForwardModelFromConfig<thrust::complex<double>, numeric::BiCGSTAB>(
                modelConfig, path, cublasHandle, cudaStream);
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
