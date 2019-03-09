#ifndef MNIST_PARSER
#define MNIST_PARSER

#include "matrix.h"
#include <cinttypes>
#include <vector>

using ValueType = uint_fast8_t;

std::vector<Matrix<ValueType, 28, 28>> parseMnistImages(const std::string& path);
std::vector<ValueType> parseMnistLabels(const std::string& path);

#endif
