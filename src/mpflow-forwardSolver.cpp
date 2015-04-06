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

    cudaSetDevice(argc <= 2 ? 0 : cudaSetDevice(atoi(argv[2])));
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
    auto const source = createSourceFromConfig<dataType>(modelConfig["source"], electrodes, cudaStream);

    time.restart();
    str::print("----------------------------------------------------");

    // load mesh from config
    auto const mesh = createMeshFromConfig(modelConfig["mesh"], path, electrodes);

    // load predefined material distribution from file, if path is given
    auto const material = [=](json_value const& materialFile) {
        if (materialFile.type != json_none) {
            return numeric::Matrix<dataType>::loadtxt(str::format("%s/%s")(path, std::string(materialFile)), cudaStream);
        }
        else {
            return std::make_shared<numeric::Matrix<dataType>>(mesh->elements.rows(), 1, cudaStream, dataType(1));
        }
    }(modelConfig["gammaFile"]);

    str::print("Mesh loaded with", mesh->nodes.rows(), "nodes and", mesh->elements.rows(), "elements");
    str::print("Time:", time.elapsed() * 1e3, "ms");

    str::print("----------------------------------------------------");
    str::print("Initialize mpFlow forward solver");
    time.restart();

    // Create main model class
    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, false>>(
        mesh, source->electrodes, modelConfig["referenceValue"].u.dbl, cudaStream);

    // Create forward solver and solve potential
    auto forwardSolver = std::make_shared<EIT::ForwardSolver<numeric::BiCGSTAB, decltype(equation)::element_type>>(
        equation, source, modelConfig["componentsCount"].u.integer, cublasHandle, cudaStream);

    // use override initial guess of potential for fixed sources to improve convergence
    if (source->type == FEM::SourceDescriptor<dataType>::Type::Fixed) {
        for (auto phi : forwardSolver->phi) {
            phi->fill(dataType(1), cudaStream);
        }
    }

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
    str::print(*result);
    result->savetxt(str::format("%s/result.txt")(path));
    potential->savetxt(str::format("%s/potential.txt")(path));

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
