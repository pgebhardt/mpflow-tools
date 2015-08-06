#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"

using namespace mpFlow;

int main(int argc, char* argv[]) {
    HighPrecisionTime time;

    // print out mpFlow version for refernce
    str::print("mpFlow version:", version::getVersionString());

    // init cuda
    cudaStream_t const cudaStream = []{ cudaStream_t stream; cudaStreamCreate(&stream); return stream; }();
    cublasHandle_t const cublasHandle = []{ cublasHandle_t handle; cublasCreate(&handle); return handle; }();

    // solve complete forward problem for a common ECT measurement setup for increasing
    // mesh density
    double density = 1.0;
    for (unsigned i = 0; i < 16; ++i) {
        // Create Mesh using libdistmesh
        time.restart();
        str::print("----------------------------------------------------");
        str::print("Create mesh with density:", density);

        auto const dist_mesh = distmesh::distmesh(distmesh::distanceFunction::circular(1.0),
            density, 1.0, 1.1 * distmesh::utils::boundingBox(2));

        str::print("Mesh created with", std::get<0>(dist_mesh).rows(), "nodes and",
            std::get<1>(dist_mesh).rows(), "element(s)");
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // update density
        density /= std::sqrt(2.0);

        // create mpflow mesh object
        auto const mesh = std::make_shared<numeric::IrregularMesh>(std::get<0>(dist_mesh),
            std::get<1>(dist_mesh));

        // create electrodes
        auto const electrodes = FEM::Ports::circularBoundary(
            16, 0.03, 0.1, mesh, 0.0);

        // create pattern
        auto const drivePattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);
        auto const measurementPattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);

        // create sources
        auto const sources = std::make_shared<FEM::Sources<float>>(
            FEM::Sources<float>::Type::Fixed, 1.0, electrodes,
            drivePattern, measurementPattern, cudaStream);

        // Create forward solver and solve potential
        time.restart();
        str::print("--------------------------");
        str::print("Solve electrical potential for all excitations");

        auto const forwardModel = std::make_shared<models::EIT<numeric::BiCGSTAB>>(
            mesh, sources, 1.0, 1.0, 1, cublasHandle, cudaStream);
        auto const gamma = std::make_shared<numeric::Matrix<float>>(mesh->elements.rows(), 1,
            cudaStream);

        time.restart();
        unsigned steps = 0;
        forwardModel->solve(gamma, cublasHandle, cudaStream, &steps);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps);
    }

    return EXIT_SUCCESS;
}
