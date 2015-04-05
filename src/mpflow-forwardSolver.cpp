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

    // check command line arguments
    if (argc <= 1) {
        str::print("You need to give a path to the model config");
        return EXIT_FAILURE;
    }

    // print out basic system info for reference
    str::print("----------------------------------------------------");
    str::print("mpFlow-forwardSolver");
    str::print("Config file:", argv[1]);
    str::print("----------------------------------------------------");
    str::print("mpFlow Version:", version::getVersionString());
    str::print("Compiler:", getCompilerName());
    str::print("Build At:", __DATE__, __TIME__);

    // print out basic cuda info for reference
    str::print("----------------------------------------------------");
    str::print("CUDA Device Info:");
    printCudaDeviceProperties();

    // init cuda
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;

    cudaSetDevice(argc <= 2 ? 0 : cudaSetDevice(atoi(argv[2])));
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // extract filename and its path from command line arguments
    std::string filename = argv[1];
    auto filenamePos = filename.find_last_of("/");
    std::string path = filenamePos == std::string::npos ? "./" : filename.substr(0, filenamePos);

    // load config from file
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

    // Create model helper classes
    auto electrodes = createBoundaryDescriptorFromConfig(modelConfig["electrodes"], modelConfig["mesh"]["radius"].u.dbl);
    auto source = createSourceFromConfig<dataType>(modelConfig["source"], electrodes, cudaStream);

    time.restart();
    str::print("----------------------------------------------------");

    // load mesh from config
    auto mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes);

    // load predefined gamma distribution from file, if path is given
    std::shared_ptr<numeric::Matrix<dataType>> gamma = nullptr;
    if (modelConfig["gammaFile"].type != json_none) {
        gamma = numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(modelConfig["gammaFile"])), cudaStream);
    }
    else {
        gamma = std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), 1, cudaStream, dataType(1));
    }

    str::print("Mesh loaded with", mesh->nodes.rows(), "nodes and", mesh->elements.rows(), "elements");
    str::print("Time:", time.elapsed() * 1e3, "ms");

    str::print("----------------------------------------------------");
    str::print("Initialize mpFlow forward solver");
    time.restart();

    // Create main model class
    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, false>>(
        mesh, source->electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    // Create forward solver and solve potential
    // use different numeric solver for different source types
    std::shared_ptr<numeric::Matrix<dataType> const> result = nullptr, potential = nullptr;
    unsigned steps = 0;
    if (source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<numeric::BiCGSTAB, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        for (auto phi : forwardSolver->phi) {
            phi->fill(1.0, cudaStream);
        }

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");

        str::print("----------------------------------------------------");
        str::print("Solve electrical potential for all excitations");
        time.restart();

        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, &steps);
        potential = forwardSolver->phi[0];
    }
    else {
        auto forwardSolver = std::make_shared<EIT::ForwardSolver<numeric::ConjugateGradient, decltype(equation)::element_type>>(
            equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");

        str::print("----------------------------------------------------");
        str::print("Solve electrical potential for all excitations");
        time.restart();

        result = forwardSolver->solve(gamma, cublasHandle, cudaStream, &steps);
        potential = forwardSolver->phi[0];
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps, ", Tolerance:",
        std::numeric_limits<typename typeTraits::extractNumericalType<dataType>::type>::epsilon());

    // Print result and save results
    result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);

    str::print("----------------------------------------------------");
    str::print("Result:");
    str::print("----------------------------------------------------");
    str::print("Real Part:");
    str::print(result->toEigen().real());
    str::print("----------------------------------------------------");
    str::print("Imaginary Part:");
    str::print(result->toEigen().imag());
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
