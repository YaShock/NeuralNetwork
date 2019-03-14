#include "network.h"
#include <cassert>
#include <cmath>
#include <sstream>

template <typename... Args>
std::string STR(Args const&... args)
{
    std::ostringstream stream;
    using List = int[];
    (void)List{0, ((void)(stream << args), 0)...};

    return stream.str();
}

using DataType = Network::DataType;

double sigmoid(double x)
{
    // return std::max(0.0, x);
    return 1 / (1 + std::exp(-x));
}

double sigmoidDerivative(double x)
{
    // return x < 0 ? 0 : 1;
    return sigmoid(x) * (1 - sigmoid(x));
}

Network::Network(const std::initializer_list<size_t>& layers)
{
    std::normal_distribution<DataType> distribution(0, 1);
    std::mt19937 numGenerator;

    mLayers.resize(layers.size());
    std::copy(layers.begin(), layers.end(), mLayers.begin());
    mNumLayers = mLayers.size();
    mWeights.reserve(mNumLayers - 1);
    mBiases.reserve(mNumLayers - 1);
    mWeightDeltas.reserve(mNumLayers - 1);
    mBiasDeltas.reserve(mNumLayers - 1);

    auto rng = [&distribution, &numGenerator](double) {
        return distribution(numGenerator);
    };

    for (size_t i = 1; i < layers.size(); ++i) {
        Matrix<DataType> w(mLayers[i], mLayers[i - 1]);
        Matrix<DataType> wDelta(mLayers[i], mLayers[i - 1]);
        w.apply(rng);
        // w.fill(0.5);
        mWeights.push_back(w);
        mWeightDeltas.push_back(wDelta);

        // std::cout << "Weight " << w << std::endl;

        ColVec<DataType> b(mLayers[i]);
        ColVec<DataType> bDelta(mLayers[i]);
        b.apply(rng);
        // b.fill(0.5);
        mBiases.push_back(b);
        mBiasDeltas.push_back(bDelta);

        // std::cout << "Bias " << b << std::endl;
    }
}

Network::Network(const std::vector<Matrix<DataType>>& weights,
                 const std::vector<ColVec<DataType>>& biases)
{
    mNumLayers = weights.size() + 1;

    mLayers.resize(mNumLayers);
    mWeightDeltas.reserve(mNumLayers - 1);
    mBiasDeltas.reserve(mNumLayers - 1);
    mWeights = weights;
    mBiases = biases;

    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        size_t layerSize = weights[i].cols();
        mLayers[i] = layerSize;

        Matrix<DataType> wDelta(mWeights[i].rows(), mWeights[i].cols());
        ColVec<DataType> bDelta(mBiases[i].rows(), mBiases[i].cols());

        mWeightDeltas.push_back(wDelta);
        mBiasDeltas.push_back(bDelta);
    }
    mLayers.back() = mWeights.back().rows();
}

#define LOG         \
    if (verboseLog) \
    std::cout

void Network::train(const Matrix<DataType>& samples, const Matrix<DataType>& labels, size_t batchSize, size_t epochs, double learningRate,
                    const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels)
{
    LOG << "Layer sizes:\n";
    for (auto layerSize : mLayers) {
        LOG << layerSize << " ";
    }
    LOG << "\n";
    LOG << "Train\n";
    if (samples.cols() != labels.cols()) {
        throw std::invalid_argument(STR("Network train: number of samples and number of labels mismatch ", samples.cols(), " ", labels.cols()));
    }
    if (samples.rows() != mLayers[0]) {
        throw std::invalid_argument("Network train: input size and first layer size mismatch");
    }
    if (labels.rows() != mLayers.back()) {
        throw std::invalid_argument("Network train: labels size and last layer size mismatch");
    }
    size_t numBatches = samples.cols() / batchSize;
    // std::cout << "batchSize: " << batchSize << std::endl;
    // std::cout << "numBatches: " << numBatches << std::endl;

    size_t img = 789;
    Matrix<double> firstCol = testSamples.col(img);
    Matrix<char> image(28, 28);
    size_t counter = 0;
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            image(i, j) = firstCol[counter] > 0 ? 'o' : ' ';
            ++counter;
        }
    }
    std::cout << "Image\n"
              << image << std::endl;
    std::cout << "label\n";
    std::cout << testLabels.col(img) << std::endl;

    for (size_t i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < numBatches; ++j) {
            // std::cout << "Batch " << j + 1 << std::endl;

            // Matrix<DataType> batch(samples.rows(), batchSize);
            // for (size_t r = 0; r < samples.rows(); ++r) {
            //     for (size_t c = 0; c < batchSize; ++c) {
            //         batch(r, c) = samples(batchSize * j + r, c);
            //     }
            // }
            // if (verboseLog)
            // std::cout << "Batch: " << batch.rows() << "X" << batch.cols() << "\n";
            // Matrix<DataType> batchLabels(labels.rows(), batchSize);
            // for (size_t r = 0; r < labels.rows(); ++r) {
            //     for (size_t c = 0; c < batchSize; ++c) {
            //         batchLabels(r, c) = labels(batchSize * j + r, c);
            //     }
            // }
            // if (verboseLog)
            // std::cout << "Expected: " << batchLabels.rows() << "X" << batchLabels.cols() << std::endl;

            std::vector<std::vector<Matrix<DataType>>> batchDeltasW(batchSize);
            std::vector<std::vector<ColVec<DataType>>> batchDeltasB(batchSize);

            for (size_t k = 0; k < batchSize; ++k) {
                if (verboseLog)
                    std::cout << "Sample: " << j * batchSize + k << std::endl;
                backpropagation(samples.col(j * batchSize + k),
                                labels.col(j * batchSize + k));
                batchDeltasW[k] = mWeightDeltas;
                batchDeltasB[k] = mBiasDeltas;
                // break;
            }

            updateWeights(batchSize, learningRate, batchDeltasW, batchDeltasB);
            // break;
        }
        std::cout << "Epoch " << i + 1 << ": ";
        validate(testSamples, testLabels);
        // break;
    }
}

void Network::validate(const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels)
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
}

Matrix<DataType> Network::predict(const Matrix<DataType>& sample)
{
    return feedforward(sample);
}

Matrix<DataType> Network::feedforward(Matrix<DataType> input)
{
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        input = activation((mWeights[i] * input).addColVec(mBiases[i]));
    }
    return input;
}

void Network::backpropagation(const ColVec<DataType>& sample, const ColVec<DataType>& label)
{
    LOG << "Label: " << label.transpose() << std::endl;
    // We don't need the first Z, but it will simplify indexing
    std::vector<Matrix<DataType>> z(mNumLayers);
    std::vector<Matrix<DataType>> a(mNumLayers);
    // Feed forward
    LOG << "Feed forward" << std::endl;
    a[0] = sample;
    for (size_t i = 1; i < mNumLayers; ++i) {
        LOG << "Layer " << i << std::endl;
        z[i] = (mWeights[i - 1] * a[i - 1]).addColVec(mBiases[i - 1]);
        // LOG << "Weights(" << i - 1 << "): " << mWeights[i - 1].rows() << "X" << mWeights[i - 1].cols() << "\n"
        // << mWeights[i - 1] << std::endl;
        LOG << "Z[i]: " << z[i].rows() << "X" << z[i].cols() << ": " << z[i].transpose() << std::endl;
        a[i] = activation(z[i]);
        LOG << "a[i]: " << a[i].rows() << "X" << a[i].cols() << ": " << a[i].transpose() << std::endl;
    }

    // Index of last layer
    size_t L = mNumLayers - 1;
    auto& output = a[L];
    assert(output.all([](double x) { return std::isfinite(x); }));

    // Backpropagation
    LOG << "Backpropagation\n";
    LOG << "output: " << output.transpose() << std::endl;
    LOG << "expected: " << label.transpose() << std::endl;
    Matrix<DataType> delta = costDerivative(output, label).hamadard(activationDerivative(z[L]));
    LOG << "deltaL: " << delta.transpose() << std::endl;

    mBiasDeltas[L - 1] = delta;
    LOG << "mBiasDeltas[L - 1]: " << mBiasDeltas[L - 1].transpose() << std::endl;
    mWeightDeltas[L - 1] = delta * (a[L - 1].transpose());
    LOG << "mWeightDeltas[L - 1]: " << mWeightDeltas[L - 1].rows() << "X" << mWeightDeltas[L - 1].cols() << "\n"
        << mWeightDeltas[L - 1] << std::endl;

    // Update deltas
    LOG << "Update deltas" << std::endl;
    for (size_t i = L - 1; i > 0; --i) {
        LOG << "Layer " << i << std::endl;
        delta = (mWeights[i].transpose() * delta).hamadard(activationDerivative(z[i]));
        LOG << "Delta: " << delta.rows() << "X" << delta.cols() << "\n"
            << delta.transpose() << std::endl;
        mBiasDeltas[i - 1] = delta;
        LOG << "mBiasDeltas(" << i - 1 << "): " << mBiasDeltas[i - 1].rows() << "X" << mBiasDeltas[i - 1].cols() << "\n"
            << mBiasDeltas[i - 1].transpose() << std::endl;
        mWeightDeltas[i - 1] = delta * (a[i - 1].transpose());
        LOG << "mWeightDeltas(" << i - 1 << "): " << mWeightDeltas[i - 1].rows() << "X" << mWeightDeltas[i - 1].cols() << "\n"
            << mWeightDeltas[i - 1] << std::endl;
    }
}

void Network::updateWeights(size_t batchSize, double eta, const std::vector<std::vector<Matrix<DataType>>>& batchDeltasW,
                            const std::vector<std::vector<ColVec<DataType>>>& batchDeltasB)
{
    LOG << "updateWeights\n";
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        LOG << "Layer " << i << std::endl;
        auto& w = mWeights[i];
        auto& b = mBiases[i];

        Matrix<DataType> wDeltasSum(w.rows(), w.cols(), 0);
        ColVec<DataType> bDeltasSum(b.rows(), 0);

        for (size_t j = 0; j < batchSize; ++j) {
            wDeltasSum += batchDeltasW[j][i];
            bDeltasSum += batchDeltasB[j][i];
        }
        // std::cout << "wDelta: " << mWeightDeltas[i] << std::endl;
        // std::cout << "bDelta: " << mBiasDeltas[i] << std::endl;
        // LOG << "old w: " << w << std::endl;
        // LOG << "old b: " << b.transpose() << std::endl;
        // Matrix<DataType> wDeltaSum(1, mWeightDeltas[i].cols(), 0);
        // Matrix<DataType> bDeltaSum(1, mBiasDeltas[i].cols(), 0);
        // for (size_t r = 0; r < mBiasDeltas[i].rows(); ++r) {
        //     for (size_t c = 0; c < mBiasDeltas[i].cols(); ++c) {
        //         wDeltaSum(0, c) = mWeightDeltas[i](r, c);
        //         bDeltaSum(0, c) = mBiasDeltas[i](r, c);
        //     }
        // }
        w = w - wDeltasSum * (eta / (double)batchSize);
        b = b - bDeltasSum * (eta / (double)batchSize);
        // w = w - (wDeltasSum);
        // b = b - (bDeltasSum);
        LOG << "new w: " << w << std::endl;
        LOG << "new b: " << b.transpose() << std::endl;
    }
}

Matrix<DataType> Network::activation(const Matrix<DataType>& input)
{
    Matrix<DataType> res(input.rows(), input.cols());
    for (size_t r = 0; r < input.rows(); ++r) {
        for (size_t c = 0; c < input.cols(); ++c) {
            res(r, c) = sigmoid(input(r, c));
        }
    }
    return res;
}

Matrix<DataType> Network::activationDerivative(const Matrix<DataType>& input)
{
    Matrix<DataType> res(input.rows(), input.cols());
    for (size_t r = 0; r < input.rows(); ++r) {
        for (size_t c = 0; c < input.cols(); ++c) {
            res(r, c) = sigmoidDerivative(input(r, c));
        }
    }
    return res;
}

ColVec<DataType> Network::costDerivative(const ColVec<DataType>& output, const ColVec<DataType>& expected)
{
    // std::cout << "Output: " << output.transpose() << std::endl;
    // std::cout << "Expected: " << expected.transpose() << std::endl;
    auto res = output - expected;
    // // std::cout << "Res: " << res.transpose() << std::endl;
    // double sum = 0;
    // for (size_t i = 0; i < res.rows(); ++i) {
    //     sum += res[i] * res[i];
    // }
    // assert(output.allFinite());
    // assert(expected.allFinite());
    // sum = std::sqrt(sum);
    // std::cout << "Cost: " << sum << std::endl;
    return res;
}
