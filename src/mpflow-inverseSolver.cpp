#include <fstream>
#include <distmesh/distmesh.h>

#include "json.h"
#include <mpflow/mpflow.h>

#include "stringtools/all.hpp"
#include "high_precision_time.h"
#include "helper.h"

using namespace mpFlow;

template <
    class forwardModelType,
    template <class> class numericalSolverType
>
void solveInverseModelFromConfig(int argc, char* argv[], json_value const& config,
    cublasHandle_t const cublasHandle, cudaStream_t const cudaStream) {
    typedef typename forwardModelType::dataType dataType;
    HighPrecisionTime time;

    // extract path from command line arguments
    std::string const filename = argv[1];
    auto const filenamePos = filename.find_last_of("/");
    std::string const path = filenamePos == std::string::npos ? "./" : filename.substr(0, filenamePos);

    // load measurement and reference data
    std::shared_ptr<numeric::Matrix<dataType>> reference = nullptr, measurement = nullptr;
    if (config["model"]["mwi"].type != json_none) {
        unsigned const frequencyIndex = argc > 4 ? atoi(argv[4]) : 85;
        bool const useReflectionParameter = argc > 5 ? (atoi(argv[5]) < 1 ? false : true) : true;

        reference = loadMWIMeasurement<dataType>(argv[2], frequencyIndex, useReflectionParameter, cudaStream);
        measurement = loadMWIMeasurement<dataType>(argv[3], frequencyIndex, useReflectionParameter, cudaStream);
    }
    else {
        if (argc == 3) {
            measurement = numeric::Matrix<dataType>::loadtxt(argv[2], cudaStream);
            reference = std::make_shared<numeric::Matrix<dataType>>(measurement->rows,
                measurement->cols, cudaStream);
        }
        else {
            reference = numeric::Matrix<dataType>::loadtxt(argv[2], cudaStream);
            measurement = numeric::Matrix<dataType>::loadtxt(argv[3], cudaStream);
        }
    }

    // read out amount of repetitions for benchmark purpose
    auto const repetitions = std::max(1, (int)config["repetitions"].u.integer);
    if (repetitions > 1) {
        str::print("----------------------------------------------------");
        str::print("Repeate reconstruction", repetitions, "times");
    }

    for (unsigned repetition = 0; repetition < repetitions; ++repetition) {
        if (repetitions > 1) {
            str::print("----------------------------------------------------");
            str::print("Repetition", repetition + 1);
        }

        str::print("----------------------------------------------------");
        str::print("Create and initialize solver");
        time.restart();

        // create solver according to json config
        auto const solver = solver::Solver<forwardModelType, numericalSolverType>::fromConfig(
            config, cublasHandle, cudaStream, path);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // copy measurement and reference data to solver
        for (auto mes : solver->measurement) {
            mes->copy(measurement, cudaStream);
        }
        for (auto cal : solver->calculation) {
            cal->copy(reference, cudaStream);
        }

        // read out amount of newton and numerical solver steps for reconstruction
        auto const maxIterations = config["solver"]["maxIterations"].u.integer;
        auto const newtonSteps = std::max(1, (int)config["solver"]["steps"].u.integer);

        // correct measurement data to enable for absolute reconstruction
        if (newtonSteps > 1) {
            reference->scalarMultiply(-1.0, cudaStream);
            solver->measurement[0]->add(reference, cudaStream);
            solver->measurement[0]->add(solver->forwardModel->result, cudaStream);
            solver->calculation[0]->copy(solver->forwardModel->result, cudaStream);
        }

        str::print("----------------------------------------------------");
        str::print("Reconstruct image");

        // reconstruct the image
        for (unsigned step = 0; step < newtonSteps; ++step) {
            str::print("----------------------------------------------------");
            str::print("Step:", step + 1);

            // restart execution timer
            cudaStreamSynchronize(cudaStream);
            time.restart();

            // do one newton iteration
            unsigned iterations = 0;
            if (step == 0) {
                solver->solveDifferential(cublasHandle, cudaStream, maxIterations, &iterations);
            }
            else {
                solver->solveAbsolute(cublasHandle, cudaStream, maxIterations, &iterations);
            }

            // get execution time for the reconstruction
            cudaStreamSynchronize(cudaStream);
            str::print("Time:", time.elapsed() * 1e3, "ms, Iterations:", iterations);

            // print current optimization norm
            solver->measurement[0]->copyToHost(cudaStream);
            solver->calculation[0]->copyToHost(cudaStream);
            cudaStreamSynchronize(cudaStream);
            str::print("Optimization norm:", sqrt((solver->measurement[0]->toEigen() - solver->calculation[0]->toEigen()).abs().square().sum()));

            // save reconstruction to file
            solver->materialDistribution->copyToHost(cudaStream);
            cudaStreamSynchronize(cudaStream);
            solver->materialDistribution->savetxt(str::format("%s/%s")(
                path, getReconstructionFileName(argc, argv, config, step)));

            // calculate some material metrices
            auto const metrices = calculateMaterialMetrices<typename typeTraits::extractNumericalType<dataType>::type>(solver->materialDistribution->toEigen(),
                solver->forwardModel->mesh);
            str::print("Material norms:", "max:", std::get<0>(metrices), "min:", std::get<1>(metrices),
                "mean:", std::get<2>(metrices), "standard deviation:", std::get<3>(metrices));

        }

        if (newtonSteps > 1) {
            // calculate final optimization norm
            solver->forwardModel->solve(solver->materialDistribution, cublasHandle, cudaStream);
            solver->forwardModel->result->copyToHost(cudaStream);
            cudaStreamSynchronize(cudaStream);

            str::print("----------------------------------------------------");
            str::print("Final optimization norm:", sqrt((solver->measurement[0]->toEigen() - solver->forwardModel->result->toEigen()).abs().square().sum()));
        }
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
    if (modelConfig["precision"].type == json_string && std::string(modelConfig["precision"]) == "single") {
        if (modelConfig["jacobian"].type == json_none) {
            if (modelConfig["mwi"].type != json_none) {
                solveInverseModelFromConfig<
                    models::MWI<numeric::CPUSolver, FEM::Equation<thrust::complex<float>, FEM::basis::Edge>>,
                    numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
            }
            else {
                if (std::string(modelConfig["source"]["type"]) == "voltage") {
                    if ((std::string(modelConfig["numericType"]) == "complex") || (std::string(modelConfig["numericType"]) == "halfComplex")) {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::BiCGSTAB, FEM::Equation<thrust::complex<float>, FEM::basis::Linear>>,
                            numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                    else {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::BiCGSTAB, FEM::Equation<float, FEM::basis::Linear>>,
                            numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                }
                else {
                    if ((std::string(modelConfig["numericType"]) == "complex") || (std::string(modelConfig["numericType"]) == "halfComplex")) {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::ConjugateGradient, FEM::Equation<thrust::complex<float>, FEM::basis::Linear>>,
                            numeric::ConjugateGradient>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                    else {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::ConjugateGradient, FEM::Equation<float, FEM::basis::Linear>>,
                            numeric::ConjugateGradient>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                }
            }
        }
        else {
            if (std::string(modelConfig["numericType"]) == "complex") {
                solveInverseModelFromConfig<models::Constant<thrust::complex<float>>, numeric::BiCGSTAB>(
                    argc, argv, *config, cublasHandle, cudaStream);
            }
            else {
                solveInverseModelFromConfig<models::Constant<float>, numeric::BiCGSTAB>(
                    argc, argv, *config, cublasHandle, cudaStream);
            }
        }
    }
    else {
        if (modelConfig["jacobian"].type == json_none) {
            if (modelConfig["mwi"].type != json_none) {
                solveInverseModelFromConfig<
                    models::MWI<numeric::CPUSolver, FEM::Equation<thrust::complex<double>, FEM::basis::Edge>>,
                    numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
            }
            else {
                if (std::string(modelConfig["source"]["type"]) == "voltage") {
                    if ((std::string(modelConfig["numericType"]) == "complex") || (std::string(modelConfig["numericType"]) == "halfComplex")) {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::BiCGSTAB, FEM::Equation<thrust::complex<double>, FEM::basis::Linear>>,
                            numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                    else {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::BiCGSTAB, FEM::Equation<double, FEM::basis::Linear>>,
                            numeric::BiCGSTAB>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                }
                else {
                    if ((std::string(modelConfig["numericType"]) == "complex") || (std::string(modelConfig["numericType"]) == "halfComplex")) {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::ConjugateGradient, FEM::Equation<thrust::complex<double>, FEM::basis::Linear>>,
                            numeric::ConjugateGradient>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                    else {
                        solveInverseModelFromConfig<
                            models::EIT<numeric::ConjugateGradient, FEM::Equation<double, FEM::basis::Linear>>,
                            numeric::ConjugateGradient>(argc, argv, *config, cublasHandle, cudaStream);
                    }
                }
            }
        }
        else {
            if (std::string(modelConfig["numericType"]) == "complex") {
                solveInverseModelFromConfig<models::Constant<thrust::complex<double>>, numeric::BiCGSTAB>(
                    argc, argv, *config, cublasHandle, cudaStream);
            }
            else {
                solveInverseModelFromConfig<models::Constant<double>, numeric::BiCGSTAB>(
                    argc, argv, *config, cublasHandle, cudaStream);
            }
        }

    }

    // cleanup
    json_value_free(config);
    cublasDestroy(cublasHandle);
    cudaStreamDestroy(cudaStream);

    return EXIT_SUCCESS;
}
