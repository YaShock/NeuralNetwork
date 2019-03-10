#include "network.h"
#include <cmath>

using DataType = Network::DataType;

double sigmoid(double x)
{
    return 1 / (1 + std::exp(-x));
}

double sigmoidDerivative(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}

Network::Network(const std::initializer_list<size_t>& layers)
{
    std::normal_distribution<DataType> distribution(0.5, 0.1);
    std::mt19937 numGenerator;

    mLayers.resize(layers.size());
    std::copy(layers.begin(), layers.end(), mLayers.begin());
    mNumLayers = mLayers.size();
    mWeights.reserve(layers.size() - 1);
    mBiases.reserve(layers.size() - 1);

    for (size_t i = 1; i < layers.size(); ++i) {
        Matrix<DataType> w(mLayers[i], mLayers[i - 1]);
        Matrix<DataType> wDelta(mLayers[i], mLayers[i - 1]);
        w.randomize(distribution, numGenerator);
        mWeights.push_back(w);
        mWeightDeltas.push_back(wDelta);

        Matrix<DataType> b(mLayers[i], 1);
        Matrix<DataType> bDelta(mLayers[i], 1);
        b.randomize(distribution, numGenerator);
        mBiases.push_back(b);
        mBiasDeltas.push_back(bDelta);
    }
}

void Network::train(const Matrix<DataType>& samples, const Matrix<DataType>& labels, size_t batchSize, size_t epochs, double learningRate)
{
    if (samples.rows() != labels.size()) {
        throw std::invalid_argument("Network train: number of samples and number of labels mismatch");
    }
    if (samples.cols() != mLayers[0]) {
        throw std::invalid_argument("Network train: input size and first layer size mismatch");
    }
    if (labels.cols() != mLayers.back()) {
        throw std::invalid_argument("Network train: labels size and last layer size mismatch");
    }
    size_t numBatches = samples.rows() / batchSize;
    for (size_t i = 0; i < epochs; ++i) {
        for (size_t j = 0; j < numBatches; ++j) {
            Matrix<DataType> batch(batchSize, samples.cols());
            for (size_t r = 0; r < batchSize; ++r) {
                for (size_t c = 0; c < samples.cols(); ++c) {
                    batch(r, c) = samples(batchSize * j + r, c);
                }
            }
            Matrix<DataType> label(batchSize, labels.cols());
            for (size_t r = 0; r < batchSize; ++r) {
                for (size_t c = 0; c < labels.cols(); ++c) {
                    batch(r, c) = labels(batchSize * j + r, c);
                }
            }
            backpropagation(batch, label);
            updateWeights(batchSize, learningRate);
        }
    }
}

Matrix<DataType> Network::predict(const Matrix<DataType>& samples)
{
    return feedforward(samples);
}

Matrix<DataType> Network::feedforward(Matrix<DataType> input)
{
    for (size_t i = 0; i < mLayers.size(); ++i) {
        input = activation(mWeights[i] * input + mBiases[i]);
    }
    return input;
}

void Network::backpropagation(const Matrix<DataType>& sample, const Matrix<DataType>& label)
{
    std::vector<Matrix<DataType>> z(mNumLayers - 1);
    std::vector<Matrix<DataType>> a(mNumLayers - 1);
    // Feed forward
    Matrix<DataType> output = sample;
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        z[i] = mWeights[i] * output + mBiases[i];
        output = activation(z[i]);
        a[i] = output;
    }
    // Backpropagation
    Matrix<DataType> delta = costDerivative(output, label).hamadard(activation(z[mNumLayers - 1]));

    mBiasDeltas[mNumLayers - 1] = delta;
    mWeightDeltas[mNumLayers - 1] = a[mNumLayers - 2] * delta;

    // Update deltas
    for (size_t i = 2; i < mNumLayers - 1; ++i) {
        delta = (mWeights[mNumLayers - i].transpose() * delta).hamadard(z[mNumLayers - i - 1]);
        mBiasDeltas[mNumLayers - i] = delta;
        mWeightDeltas[mNumLayers - i] = a[mNumLayers - i - 1] * delta;
    }
}

void Network::updateWeights(size_t batchSize, double eta)
{
    for (size_t i = 0; i < mNumLayers - 1; ++i) {
        auto& w = mWeights[i];
        auto& b = mBiases[i];
        Matrix<DataType> wDeltaSum(1, mWeightDeltas[i].cols(), 0);
        Matrix<DataType> bDeltaSum(1, mBiasDeltas[i].cols(), 0);
        for (size_t r = 0; r < mBiasDeltas[i].rows(); ++r) {
            for (size_t c = 0; c < mBiasDeltas[i].cols(); ++c) {
                wDeltaSum(0, c) = mWeightDeltas[i](r, c);
                bDeltaSum(0, c) = mBiasDeltas[i](r, c);
            }
        }
        w = w - wDeltaSum * (eta / (double)batchSize);
        b = b - bDeltaSum * (eta / (double)batchSize);
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

Matrix<DataType> Network::costDerivative(const Matrix<DataType>& output, const Matrix<DataType>& expected)
{
    return output - expected;
}
