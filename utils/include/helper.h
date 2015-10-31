#ifndef MPFLOW_TOOLS_UTILS_HELPER_H
#define MPFLOW_TOOLS_UTILS_HELPER_H

// load measurement data from MWI
template <class dataType>
std::shared_ptr<mpFlow::numeric::Matrix<dataType>> loadMWIMeasurement(
    std::string const filename, unsigned const frequencyIndex, bool const useReflectionParameter,
    cudaStream_t const cudaStream);

// print properties of current cuda device
void printCudaDeviceProperties();

// get basic compiler name
std::string getCompilerName();

// create proper file name for a reconstructed image
std::string getReconstructionFileName(int const argc, char* const argv[], json_value const& config,
    unsigned const iteration);

// calculate some metrices for material distribution
template <class dataType>
std::tuple<dataType, dataType, dataType, dataType> calculateMaterialMetrices(
    Eigen::Ref<Eigen::Array<dataType, Eigen::Dynamic, Eigen::Dynamic> const> const materialDistribution,
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const mesh) {
    // calc area of each element in mesh
    Eigen::Array<dataType, Eigen::Dynamic, 1> area(mesh->elements.rows());
    for (unsigned e = 0; e < mesh->elements.rows(); ++e) {
        auto const nodes = mesh->elementNodes(e);

        area(e) = 0.5 * std::abs(
            (nodes(1, 0) - nodes(0, 0)) * (nodes(2, 1) - nodes(0, 1)) -
            (nodes(2, 0) - nodes(0, 0)) * (nodes(1, 1) - nodes(0, 1)));
    }

    // calculate mean value and standard deviation
    auto const mean = (materialDistribution * area).sum() / area.sum();
    auto const standardDeviation = sqrt(((materialDistribution - mean).square() * area).sum() / area.sum());

    return std::make_tuple(
        materialDistribution.maxCoeff(),
        materialDistribution.minCoeff(),
        mean, standardDeviation);

}

template <class dataType>
std::tuple<std::complex<dataType>, std::complex<dataType>, std::complex<dataType>, std::complex<dataType>> calculateMaterialMetrices(
    Eigen::Ref<Eigen::Array<std::complex<dataType>, Eigen::Dynamic, Eigen::Dynamic> const> const materialDistribution,
    std::shared_ptr<mpFlow::numeric::IrregularMesh const> const mesh) {
    // calc area of each element in mesh
    Eigen::Array<std::complex<dataType>, Eigen::Dynamic, 1> area(mesh->elements.rows());
    for (unsigned e = 0; e < mesh->elements.rows(); ++e) {
        auto const nodes = mesh->elementNodes(e);

        area(e) = 0.5 * std::abs(
            (nodes(1, 0) - nodes(0, 0)) * (nodes(2, 1) - nodes(0, 1)) -
            (nodes(2, 0) - nodes(0, 0)) * (nodes(1, 1) - nodes(0, 1)));
    }

    // calculate mean value and standard deviation
    auto const mean = (materialDistribution * area).sum() / area.sum();
    auto const standardDeviation = sqrt(((materialDistribution - mean).square() * area).sum() / std::complex<dataType>(area.sum()));

    return std::make_tuple(
        std::complex<dataType>(materialDistribution.real().maxCoeff(), materialDistribution.imag().maxCoeff()),
        std::complex<dataType>(materialDistribution.real().minCoeff(), materialDistribution.imag().minCoeff()),
        mean, standardDeviation);
}

#endif
