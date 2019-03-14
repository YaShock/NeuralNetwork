#include "matrix.h"
#include <vector>

double sigmoid(double x);
double sigmoidDerivative(double x);

class Network
{
public:
    using DataType = double;

    // First is input layer, last is output layer
    // TODO: activation function as parameter
    explicit Network(const std::initializer_list<size_t>& layers);
    Network(const std::vector<Matrix<DataType>>& weights,
            const std::vector<ColVec<DataType>>& biases);

    void train(const Matrix<DataType>& samples, const Matrix<DataType>& labels, size_t batchSize, size_t epochs, double learningRate,
               const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels);
    void validate(const Matrix<DataType>& testSamples, const Matrix<DataType>& testLabels);
    Matrix<DataType> predict(const Matrix<DataType>& sample);

private:
    Matrix<DataType> feedforward(Matrix<DataType> input);
    void backpropagation(const ColVec<DataType>& sample, const ColVec<DataType>& label);
    Matrix<DataType> activation(const Matrix<DataType>& input);
    Matrix<DataType> activationDerivative(const Matrix<DataType>& input);
    ColVec<DataType> costDerivative(const ColVec<DataType>& output, const ColVec<DataType>& expected);
    void updateWeights(size_t batchSize, double learningRate, const std::vector<std::vector<Matrix<DataType>>>& batchDeltasW,
                       const std::vector<std::vector<ColVec<DataType>>>& batchDeltasB);

    std::vector<Matrix<DataType>> mWeights;
    std::vector<ColVec<DataType>> mBiases;
    std::vector<size_t> mLayers;
    size_t mNumLayers;
    std::vector<Matrix<DataType>> mWeightDeltas;
    std::vector<ColVec<DataType>> mBiasDeltas;

    bool verboseLog = false;
};
