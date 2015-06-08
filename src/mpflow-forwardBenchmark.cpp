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
    cudaStream_t cudaStream = nullptr;
    cublasHandle_t cublasHandle = nullptr;
    cublasCreate(&cublasHandle);
    cudaStreamCreate(&cudaStream);

    // solve complete forward problem for a common ECT measurement setup for increasing
    // mesh density
    double density = 1.0;
    for (unsigned i = 0; i < 16; ++i) {
        // Create Mesh using libdistmesh
        time.restart();
        str::print("----------------------------------------------------");
        str::print("Create mesh with density:", density);

        auto dist_mesh = distmesh::distmesh(distmesh::distanceFunction::circular(1.0),
            density, 1.0, 1.1 * distmesh::boundingBox(2));
        auto boundary = distmesh::boundEdges(std::get<1>(dist_mesh));

        str::print("Mesh created with", std::get<0>(dist_mesh).rows(), "nodes and",
            std::get<1>(dist_mesh).rows(), "element(s)");
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // update density
        density /= std::sqrt(2.0);

        // create mpflow mesh object
        auto mesh = std::make_shared<numeric::IrregularMesh>(std::get<0>(dist_mesh),
            std::get<1>(dist_mesh), boundary, 1.0);

        // create electrodes
        auto electrodes = FEM::BoundaryDescriptor::circularBoundary(
            16, 0.03, 0.1, 1.0, 0.0);

        // create pattern
        auto drivePattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);
        auto measurementPattern = numeric::Matrix<int>::eye(electrodes->count, cudaStream);

        // create source
        auto source = std::make_shared<FEM::SourceDescriptor<float>>(
            FEM::SourceDescriptor<float>::Type::Fixed, 1.0, electrodes,
            drivePattern, measurementPattern, cudaStream);

        // Create forward solver and solve potential
        time.restart();
        str::print("--------------------------");
        str::print("Solve electrical potential for all excitations");

        auto forwardSolver = std::make_shared<models::EIT<numeric::BiCGSTAB>>(
            mesh, source, 1.0, 1, cublasHandle, cudaStream);
        auto gamma = std::make_shared<numeric::Matrix<float>>(mesh->elements.rows(), 1,
            cudaStream);

        time.restart();
        unsigned steps = 0;
        forwardSolver->solve(gamma, cublasHandle, cudaStream, &steps);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps);
    }

    return EXIT_SUCCESS;
}
