#include "matrix.h"
#include <vector>

double sigmoid(double x);
double sigmoidPrime(double x);

double relu(double x);
double reluPrime(double x);

class Network
{
public:
    using DataType = double;

    using ActFn = DataType (*)(DataType);
    // First function is used for activation, second is it's derivative
    using Activation = std::pair<ActFn, ActFn>;
    // Some predefined activation functions/
    // Sigmoid is the default activation function
    constexpr static Activation Sigmoid = std::make_pair(&sigmoid, &sigmoidPrime);
    constexpr static Activation ReLU = std::make_pair(&relu, &reluPrime);

    // First is input layer, last is output layer
    // Activation function vector has either one function used by all layers
    // or an activation function specified for each layer except input
    explicit Network(const std::initializer_list<size_t>& layers,
                     const std::vector<Activation>& activations = {Sigmoid});
    Network(const std::vector<Matrix<DataType>>& weights,
            const std::vector<ColVec<DataType>>& biases,
            const std::vector<Activation>& activations = {Sigmoid});

    void train(const Matrix<DataType>& samples, const Matrix<DataType>& labels, size_t batchSize, size_t epochs, double learningRate,
               const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels);
    void validate(const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels);
    Matrix<DataType> predict(const Matrix<DataType>& sample);

private:
    void shuffleSamples(Matrix<DataType>& samples, Matrix<DataType>& labels);
    Matrix<DataType> feedforward(Matrix<DataType> input);
    void backpropagation(const ColVec<DataType>& sample, const ColVec<DataType>& label);
    Matrix<DataType> activation(const Matrix<DataType>& input, size_t layer);
    Matrix<DataType> activationPrime(const Matrix<DataType>& input, size_t layer);
    Activation& getActivation(size_t layer);
    ColVec<DataType> costDerivative(const ColVec<DataType>& output, const ColVec<DataType>& expected);
    void updateWeights(size_t batchSize, double learningRate, const std::vector<std::vector<Matrix<DataType>>>& batchDeltasW,
                       const std::vector<std::vector<ColVec<DataType>>>& batchDeltasB);

    std::vector<Matrix<DataType>> mWeights;
    std::vector<ColVec<DataType>> mBiases;
    std::vector<size_t> mLayers;
    size_t mNumLayers;
    std::vector<Matrix<DataType>> mWeightDeltas;
    std::vector<ColVec<DataType>> mBiasDeltas;
    std::vector<Activation> mActivations;

    bool verboseLog = false;
};
