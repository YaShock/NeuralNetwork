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

    void train(const Matrix<DataType>& samples, const Matrix<DataType>& labels, size_t batchSize, size_t epochs, double learningRate);
    Matrix<DataType> predict(const Matrix<DataType>& samples);

private:
    Matrix<DataType> feedforward(Matrix<DataType> input);
    void backpropagation(const Matrix<DataType>& sample, const Matrix<DataType>& label);
    Matrix<DataType> activation(const Matrix<DataType>& input);
    Matrix<DataType> costDerivative(const Matrix<DataType>& output, const Matrix<DataType>& expected);
    void updateWeights(size_t batchSize, double learningRate);

    std::vector<Matrix<DataType>> mWeights, mBiases;
    std::vector<size_t> mLayers;
    size_t mNumLayers;
    std::vector<Matrix<DataType>> mWeightDeltas, mBiasDeltas;
};
