#include <fstream>
#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "json.h"
#include "helper.h"

using namespace mpFlow;

template <class dataType>
dataType parseReferenceValue(json_value const& config) {
    return config.u.dbl;    
}

template <>
thrust::complex<double> parseReferenceValue(json_value const& config) {
    if (config.type == json_array) {
        return thrust::complex<double>(config[0], config[1]);
    }
    else {
        return thrust::complex<double>(config.u.dbl);
    }
}

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

    // read out reference value
    auto const referenceValue = parseReferenceValue<dataType>(modelConfig["material"]);
    
    // Create main model class
    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create main model class");

    auto equation = std::make_shared<FEM::Equation<dataType, FEM::basis::Linear, logarithmic>>(
        mesh, source->electrodes, referenceValue, cudaStream);

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    time.restart();
    str::print("----------------------------------------------------");
    str::print("Create and initialize solver");

    // extract parallel images count
    int const parallelImages = std::max(1, (int)solverConfig["parallelImages"].u.integer);

    // create inverse solver and forward model
    auto solver = std::make_shared<EIT::Solver<numericalSolverType, numericalSolverType,
        typename decltype(equation)::element_type>>(
        equation, source, std::max(1, (int)modelConfig["componentsCount"].u.integer),
        parallelImages, cublasHandle, cudaStream);
    solver->preSolve(cublasHandle, cudaStream);
    
    // extract regularization parameter
    double const regularizationFactor = solverConfig["regularizationFactor"].u.dbl;
    auto const regularizationType = [](json_value const& config) {
        typedef solver::Inverse<dataType, numericalSolverType> inverseSolverType;
        
        if (std::string(config) == "diagonal") {
            return inverseSolverType::RegularizationType::diagonal;
        }
        else if (std::string(config) == "totalVariational") {
            return inverseSolverType::RegularizationType::totalVariational;
        }
        else {
            return inverseSolverType::RegularizationType::identity;    
        }
    }(solverConfig["regularizationType"]);
    
    solver->inverseSolver->setRegularizationParameter(regularizationFactor, regularizationType,
        cublasHandle, cudaStream);

    // copy measurement and reference data to solver
    for (auto mes : solver->measurement) {
        mes->copy(measurement, cudaStream);
    }
    for (auto cal : solver->calculation) {
        cal->copy(reference, cudaStream);
    }

    cudaStreamSynchronize(cudaStream);
    str::print("Time:", time.elapsed() * 1e3, "ms");

    // read out amount of newton steps for reconstruction
    int const newtonSteps = std::max(1, (int)solverConfig["steps"].u.integer);

    str::print("----------------------------------------------------");
    str::print("Reconstruct image");

    // reconstruct image
    if (newtonSteps == 1) {
        time.restart();

        unsigned steps = 0;
        solver->solveDifferential(cublasHandle, cudaStream, 0, &steps);
    
        cudaStreamSynchronize(cudaStream);
        str::print("Time per image:", time.elapsed() * 1e3 / parallelImages, "ms, FPS:",
            parallelImages / time.elapsed(), "Hz, Iterations:", steps);        
    }
    else {
        // correct measurement
        solver->calculation[0]->scalarMultiply(-1.0, cudaStream);
        solver->measurement[0]->add(solver->calculation[0], cudaStream);
        solver->measurement[0]->add(solver->forwardSolver->result, cudaStream);    
        
        cudaStreamSynchronize(cudaStream);
        time.restart();

        solver->solveAbsolute(newtonSteps, cublasHandle, cudaStream);
        
        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");                        
        
    }

    // Save results
    solver->result->copyToHost(cudaStream);
    cudaStreamSynchronize(cudaStream);
    
    if (logarithmic) {
        auto result = referenceValue * (log(10.0) * solver->result->toEigen() / 10.0).exp();
        numeric::Matrix<dataType>::fromEigen(result, cudaStream)->savetxt(str::format("%s/reconstruction.txt")(path));        
    }
    else {
        solver->result->savetxt(str::format("%s/reconstruction.txt")(path));
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
