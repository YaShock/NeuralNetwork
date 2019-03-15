#include "matrix.h"
#include "mnist_parser.h"
#include "network.h"
#include <algorithm>
#include <exception>
#include <iostream>

struct Params {
    size_t epochs;
    size_t batchSize;
    double learningRate;
    std::vector<size_t> layers;
    std::vector<Network::Activation> activations;
};

std::ostream& operator<<(std::ostream& os, const Params& params)
{
    os << "epochs: " << params.epochs << " batchSize: " << params.batchSize
       << " learningRate: " << params.learningRate << " layers: { ";
    for (size_t layer : params.layers) {
        os << layer << " ";
    }
    os << "}";
    return os;
}

using DataType = Network::DataType;

Params findBestParams(const Matrix<DataType>& samples, const Matrix<DataType>& labels, const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels, const std::vector<Params> allParams)
{
    size_t bestIdx = 0;
    double bestScore = 0;
    for (size_t i = 0; i < allParams.size(); ++i) {
        const auto& params = allParams[i];
        Network network(params.layers);
        double score = network.train(samples, labels, params.batchSize, params.epochs, params.learningRate, testSamples, testLabels);
        std::cout << "Params(" << i + 1 << "/" << allParams.size() << "): " << params << " score: " << score << std::endl;
        if (score > bestScore) {
            bestScore = score;
            bestIdx = i;
        }
    }
    return allParams[bestIdx];
}

std::vector<Params> generateParams()
{
    std::vector<Params> allParams;

    std::vector<double> learningRates = {
        0.1, 0.5, 1};
    std::vector<size_t> batchSizes = {
        1, 10, 50, 100};
    std::vector<std::vector<Network::Activation>> activations = {
        {Network::Sigmoid}, {Network::ReLU}};
    std::vector<std::vector<size_t>> layers = {
        {784, 50, 10},
        {784, 30, 10, 10},
        {784, 50, 30, 10, 10}};

    for (double learningRate : learningRates) {
        for (size_t batchSize : batchSizes) {
            for (auto& activation : activations) {
                for (auto& layer : layers) {
                    Params params;
                    params.epochs = 1;
                    params.learningRate = learningRate;
                    params.batchSize = batchSize;
                    params.activations = activation;
                    params.layers = layer;
                    allParams.push_back(std::move(params));
                }
            }
        }
    }

    return allParams;
}

int main(int argc, char const* argv[])
{
    auto images = parseMnistImages("data/train-images.idx3-ubyte");
    auto labels = parseMnistLabels("data/train-labels.idx1-ubyte");
    auto testImages = parseMnistImages("data/t10k-images.idx3-ubyte");
    auto testLabels = parseMnistLabels("data/t10k-labels.idx1-ubyte");

    try {
        // auto params = findBestParams(images, labels, testImages, testLabels, generateParams());

        Network network({784, 50, 10});
        network.train(images, labels, 1, 10, 1, testImages, testLabels);
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
