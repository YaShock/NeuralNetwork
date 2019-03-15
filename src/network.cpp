#include "network.h"
#include <algorithm>
#include <cassert>
#include <cmath>
#include <sstream>

namespace
{

template <typename... Args>
std::string STR(Args const&... args)
{
    std::ostringstream stream;
    using List = int[];
    (void)List{0, ((void)(stream << args), 0)...};

    return stream.str();
}

} // namespace

using DataType = Network::DataType;
using Activation = Network::Activation;

double sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

double sigmoidPrime(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

double relu(double x)
{
    return std::max(0.0, x);
}

double reluPrime(double x)
{
    return x < 0 ? 0 : 1;
}

Network::Network(const std::vector<size_t>& layers,
                 const std::vector<Activation>& activations)
{
    if (activations.size() != layers.size() - 1 &&
        activations.size() != 1) {
        throw std::invalid_argument("Network ctor: number of activation functions must be 1 or equal to layers size - 1");
    }
    if (layers.size() < 3) {
        throw std::invalid_argument("Network ctor: number of layers must be at least 3");
    }
    mActivations = activations;

    std::normal_distribution<DataType> distribution(0, 1);
    std::mt19937 numGenerator;

    mNumLayers = layers.size();
    mWeights.reserve(mNumLayers - 1);
    mBiases.reserve(mNumLayers - 1);
    mWeightDeltas.reserve(mNumLayers - 1);
    mBiasDeltas.reserve(mNumLayers - 1);

    auto rng = [&distribution, &numGenerator](double) {
        return distribution(numGenerator);
    };

    for (size_t i = 1; i < layers.size(); ++i) {
        Matrix<DataType> w(layers[i], layers[i - 1]);
        Matrix<DataType> wDelta(layers[i], layers[i - 1]);
        w.apply(rng);
        mWeights.push_back(w);
        mWeightDeltas.push_back(wDelta);

        ColVec<DataType> b(layers[i]);
        ColVec<DataType> bDelta(layers[i]);
        b.apply(rng);
        mBiases.push_back(b);
        mBiasDeltas.push_back(bDelta);
    }
}

Network::Network(const std::vector<Matrix<DataType>>& weights,
                 const std::vector<ColVec<DataType>>& biases,
                 const std::vector<Activation>& activations)
{
    if (activations.size() != weights.size() &&
        activations.size() != 1) {
        throw std::invalid_argument("Network ctor: number of activation functions must be 1 or equal to number of weights");
    }
    if (weights.size() != biases.size()) {
        throw std::invalid_argument("Network ctor: number of weights and number of biases must be equal");
    }
    if (weights.size() < 2) {
        throw std::invalid_argument("Network ctor: number of weights/biases must be at least 2");
    }
    mActivations = activations;

    mNumLayers = weights.size() + 1;
    mWeightDeltas.reserve(mNumLayers - 1);
    mBiasDeltas.reserve(mNumLayers - 1);
    mWeights = weights;
    mBiases = biases;

    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        Matrix<DataType> wDelta(mWeights[i].rows(), mWeights[i].cols());
        ColVec<DataType> bDelta(mBiases[i].rows(), mBiases[i].cols());

        mWeightDeltas.push_back(wDelta);
        mBiasDeltas.push_back(bDelta);
    }
}

double Network::train(const Matrix<DataType>& samples,
                      const Matrix<DataType>& labels,
                      size_t batchSize, size_t epochs, double learningRate,
                      const Matrix<DataType>& testSamples,
                      const Matrix<DataType>& testLabels,
                      bool shuffle)
{
    if (samples.cols() != labels.cols()) {
        throw std::invalid_argument(STR("Network train: number of samples and number of labels mismatch ", samples.cols(), " ", labels.cols()));
    }
    if (samples.rows() != mWeights[0].cols()) {
        throw std::invalid_argument(STR("Network train: input size and first layer size mismatch: ", samples.rows(), " ", mWeights[0].cols()));
    }
    if (labels.rows() != mWeights.back().rows()) {
        throw std::invalid_argument("Network train: labels size and last layer size mismatch: ");
    }
    size_t numBatches = samples.cols() / batchSize;
    double score = 0;
    auto sSamples = samples;
    auto sLabels = labels;

    for (size_t i = 0; i < epochs; ++i) {
        if (shuffle) {
            shuffleSamples(sSamples, sLabels);
        }
        for (size_t j = 0; j < numBatches; ++j) {
            std::vector<std::vector<Matrix<DataType>>> batchDeltasW(batchSize);
            std::vector<std::vector<ColVec<DataType>>> batchDeltasB(batchSize);

            for (size_t k = 0; k < batchSize; ++k) {
                backpropagation(sSamples.col(j * batchSize + k),
                                sLabels.col(j * batchSize + k));
                batchDeltasW[k] = mWeightDeltas;
                batchDeltasB[k] = mBiasDeltas;
            }

            updateWeights(batchSize, learningRate, batchDeltasW, batchDeltasB);
        }
        std::cout << "Epoch " << i + 1 << ": ";
        score = validate(testSamples, testLabels);
    }
    return score;
}

void Network::shuffleSamples(Matrix<DataType>& samples, Matrix<DataType>& labels)
{
    std::vector<size_t> indeces(samples.cols());
    std::iota(indeces.begin(), indeces.end(), 0);
    std::random_shuffle(indeces.begin(), indeces.end());

    Matrix<double> shuffledSamples(samples.rows(), samples.cols());
    Matrix<double> shuffledLabels(labels.rows(), labels.cols());

    for (size_t i = 0; i < samples.cols(); ++i) {
        size_t newIdx = indeces[i];
        for (size_t j = 0; j < samples.rows(); ++j) {
            shuffledSamples(j, i) = samples(j, newIdx);
        }
        for (size_t j = 0; j < labels.rows(); ++j) {
            shuffledLabels(j, i) = labels(j, newIdx);
        }
    }
    samples = std::move(shuffledSamples);
    labels = std::move(shuffledLabels);
}

double Network::validate(const Matrix<DataType>& testSamples,
                         const Matrix<DataType>& testLabels)
{
    size_t all = testSamples.cols();
    size_t correct = 0;
    for (size_t s = 0; s < all; ++s) {
        auto output = predict(testSamples.col(s));
        size_t maxIdx = 0;
        for (size_t i = 1; i < output.rows(); ++i) {
            if (output(i, 0) > output(maxIdx, 0)) {
                maxIdx = i;
            }
        }
        if (testLabels(maxIdx, s) == 1) {
            ++correct;
        }
    }
    std::cout << correct << " / " << all << std::endl;
    return (double)correct / (double)all;
}

Matrix<DataType> Network::predict(const Matrix<DataType>& sample)
{
    return feedforward(sample);
}

Matrix<DataType> Network::feedforward(Matrix<DataType> input)
{
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        input = activation((mWeights[i] * input).addColVec(mBiases[i]), i);
    }
    return input;
}

void Network::backpropagation(const ColVec<DataType>& sample,
                              const ColVec<DataType>& label)
{
    // We don't need the first Z, but it will simplify indexing
    std::vector<Matrix<DataType>> z(mNumLayers);
    std::vector<Matrix<DataType>> a(mNumLayers);
    // Feed forward
    a[0] = sample;
    for (size_t i = 1; i < mNumLayers; ++i) {
        z[i] = (mWeights[i - 1] * a[i - 1]).addColVec(mBiases[i - 1]);
        a[i] = activation(z[i], i);
    }

    // Index of last layer
    size_t L = mNumLayers - 1;
    auto& output = a[L];

    // Backpropagation
    Matrix<DataType> delta = costDerivative(output, label).hamadard(activationPrime(z[L], L));

    mBiasDeltas[L - 1] = delta;
    mWeightDeltas[L - 1] = delta * (a[L - 1].transpose());

    // Update deltas
    for (size_t i = L - 1; i > 0; --i) {
        delta = (mWeights[i].transpose() * delta).hamadard(activationPrime(z[i], i));
        mBiasDeltas[i - 1] = delta;
        mWeightDeltas[i - 1] = delta * (a[i - 1].transpose());
    }
}

void Network::updateWeights(size_t batchSize, double eta,
                            const std::vector<std::vector<Matrix<DataType>>>& batchDeltasW,
                            const std::vector<std::vector<ColVec<DataType>>>& batchDeltasB)
{
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        auto& w = mWeights[i];
        auto& b = mBiases[i];

        Matrix<DataType> wDeltasSum(w.rows(), w.cols(), 0);
        ColVec<DataType> bDeltasSum(b.rows(), 0);

        for (size_t j = 0; j < batchSize; ++j) {
            wDeltasSum += batchDeltasW[j][i];
            bDeltasSum += batchDeltasB[j][i];
        }
        w = w - wDeltasSum * (eta / (double)batchSize);
        b = b - bDeltasSum * (eta / (double)batchSize);
    }
}

Matrix<DataType> Network::activation(const Matrix<DataType>& input, size_t layer)
{
    return input.transform(getActivation(layer).first);
}

Matrix<DataType> Network::activationPrime(const Matrix<DataType>& input, size_t layer)
{
    return input.transform(getActivation(layer).second);
}

Activation& Network::getActivation(size_t layer)
{
    return mActivations.size() == 1 ? mActivations.front() : mActivations[layer];
}

ColVec<DataType> Network::costDerivative(const ColVec<DataType>& output, const ColVec<DataType>& expected)
{
    return output - expected;
}
