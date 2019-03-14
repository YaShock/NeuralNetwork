#include "mnist_parser.h"
#include <exception>
#include <fstream>
#include <iostream>

using ValueType = uint_fast8_t;

namespace
{

int swapEndian(int i)
{
    unsigned char c1, c2, c3, c4;

    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;

    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

bool isBigEndian()
{
    union {
        uint32_t i;
        char c[4];
    } bint = {0x01020304};

    return bint.c[0] == 1;
}

int readUint32(std::istream& is)
{
    int x;
    is.read((char*)&x, sizeof(x));
    if (!isBigEndian()) {
        x = swapEndian(x);
    }
    return x;
}

} // namespace

Matrix<double> parseMnistImages(const std::string& path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::invalid_argument("File cannot be opened");
    }

    uint32_t magicNum, nImages, nRows, nCols;
    magicNum = readUint32(file);

    if (magicNum != 2051) {
        throw std::invalid_argument("Magic number is not 2051");
    }

    nImages = readUint32(file);
    nRows = readUint32(file);
    nCols = readUint32(file);

    std::cout << "NO images: " << nImages << std::endl;
    std::cout << "Rows: " << nRows << " Cols: " << nCols << std::endl;

    Matrix<double> images(nCols * nRows, nImages);

    for (uint32_t j = 0; j < nImages; ++j) {
        for (uint32_t i = 0; i < nCols * nRows; ++i) {
            ValueType val = 0;
            file.read((char*)&val, 1);
            images(i, j) = static_cast<double>(val) / 255;
        }
    }

    return images;
}

Matrix<double> parseMnistLabels(const std::string& path)
{
    std::ifstream file(path, std::ios::in | std::ios::binary);

    if (!file.is_open()) {
        throw std::invalid_argument("File cannot be opened");
    }

    uint32_t nLabels, magicNum;
    magicNum = readUint32(file);

    if (magicNum != 2049) {
        throw std::invalid_argument("Magic number is not 2049");
    }

    nLabels = readUint32(file);

    Matrix<double> labels(10, nLabels);

    std::cout << "NO labels: " << nLabels << std::endl;

    for (uint32_t i = 0; i < nLabels; ++i) {
        uint32_t val = 0;
        file.read((char*)&val, 1);
        for (uint32_t j = 0; j < 10; ++j) {
            if (val == j) {
                labels(j, i) = 1;
            } else {
                labels(j, i) = 0;
            }
        }
    }

    return labels;
}
