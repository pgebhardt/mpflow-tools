#include <distmesh/distmesh.h>
#include <mpflow/mpflow.h>
#include "stringtools/all.hpp"
#include "high_precision_time.h"

using namespace mpFlow;

// helper function to create unit matrix
template <class type>
std::shared_ptr<numeric::Matrix<type>> eye(dtype::index size, cudaStream_t cudaStream) {
    auto matrix = std::make_shared<numeric::Matrix<type>>(size, size, cudaStream);
    for (dtype::index i = 0; i < size; ++i) {
        (*matrix)(i, i) = 1;
    }
    matrix->copyToDevice(cudaStream);

    return matrix;
}

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
    dtype::real density = 1.0;
    for (dtype::index i = 0; i < 16; ++i) {
        // Create Mesh using libdistmesh
        time.restart();
        str::print("----------------------------------------------------");
        str::print("Create mesh with density:", density);

        auto dist_mesh = distmesh::distmesh(distmesh::distance_function::circular(1.0),
            density, 1.0, 1.1 * distmesh::bounding_box(2));
        auto boundary = distmesh::boundedges(std::get<1>(dist_mesh));

        str::print("Mesh created with", std::get<0>(dist_mesh).rows(), "nodes and",
            std::get<1>(dist_mesh).rows(), "element(s)");
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // update density
        density /= std::sqrt(2.0);

        // create mpflow mesh object
        auto mesh = std::make_shared<numeric::IrregularMesh>(
            numeric::matrix::fromEigen<dtype::real, double>(std::get<0>(dist_mesh), cudaStream),
            numeric::matrix::fromEigen<dtype::index, int>(std::get<1>(dist_mesh), cudaStream),
            numeric::matrix::fromEigen<dtype::index, int>(boundary, cudaStream), 1.0, 1.0);

        // create electrodes
        auto electrodes = FEM::boundaryDescriptor::circularBoundary(
            16, std::make_tuple(0.03, 0.1), 1.0, 0.0);

        // create pattern
        auto drivePattern = numeric::Matrix<dtype::real>::eye(electrodes->count, cudaStream);
        auto measurementPattern = numeric::Matrix<dtype::real>::eye(electrodes->count, cudaStream);

        // create source
        auto source = std::make_shared<FEM::SourceDescriptor>(
            FEM::SourceDescriptor::Type::Fixed, 1.0, electrodes,
            drivePattern, measurementPattern, cudaStream);

        // create equation
        time.restart();
        str::print("--------------------------");
        str::print("Create equation model class");

        auto equation = std::make_shared<FEM::Equation<dtype::real, FEM::basis::Linear>>(
            mesh, electrodes, 1.0, cudaStream);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms");

        // Create forward solver and solve potential
        str::print("--------------------------");
        str::print("Solve electrical potential for all excitations");

        auto forwardSolver = std::make_shared<EIT::ForwardSolver<FEM::basis::Linear,
            numeric::BiCGSTAB>>(equation, source, 1, cublasHandle, cudaStream);
        auto gamma = std::make_shared<numeric::Matrix<dtype::real>>(mesh->elements->rows, 1,
            cudaStream);

        time.restart();
        dtype::index steps = 0;
        forwardSolver->solve(gamma, cublasHandle, cudaStream, 1e-9, &steps);

        cudaStreamSynchronize(cudaStream);
        str::print("Time:", time.elapsed() * 1e3, "ms, Steps:", steps);
    }

    return EXIT_SUCCESS;
}
