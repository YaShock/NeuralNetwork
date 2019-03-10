#ifndef MNIST_PARSER
#define MNIST_PARSER

#include "matrix.h"
#include <cinttypes>
#include <vector>

Matrix<double> parseMnistImages(const std::string& path);
Matrix<double> parseMnistLabels(const std::string& path);

#endif
