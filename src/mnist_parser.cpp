#include "mnist_parser.h"
#include <exception>
#include <fstream>
#include <iostream>

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

// bool isBigEndian()
// {
//     union {
//         uint32_t i;
//         char c[4];
//     } bint = {0x01020304};

//     return bint.c[0] == 1;
// }

int readUint32(std::istream& is)
{
    int x;
    is.read((char*)&x, sizeof(x));
    // if (!isBigEndian()) {
    x = swapEndian(x);
    // }
    return x;
}

} // namespace

std::vector<Matrix<ValueType, 28, 28>> parseMnistImages(const std::string& path)
{
    std::vector<Matrix<ValueType, 28, 28>> images;

    std::ifstream file(path, std::ios::in | std::ios::binary);

    uint32_t magicNum, nImages, nRows, nCols;
    magicNum = readUint32(file);

    if (magicNum != 2051) {
        throw std::invalid_argument("Magic number is not 2051");
    }

    nImages = readUint32(file);
    nRows = readUint32(file);
    nCols = readUint32(file);

    images.reserve(nImages);

    for (uint32_t i = 0; i < nImages; ++i) {
        Matrix<ValueType, 28, 28> matrix;
        for (uint32_t r = 0; r < nRows; ++r) {
            for (uint32_t c = 0; c < nCols; ++c) {
                ValueType val = 0;
                file.read((char*)&val, 1);
                matrix(r, c) = val;
            }
        }
        images.push_back(matrix);
    }

    return images;
}

std::vector<ValueType> parseMnistLabels(const std::string& path)
{
    std::vector<ValueType> labels;

    std::ifstream file(path, std::ios::in | std::ios::binary);

    uint32_t nLabels, magicNum;
    magicNum = readUint32(file);

    if (magicNum != 2049) {
        throw std::invalid_argument("Magic number is not 2049");
    }

    nLabels = readUint32(file);
    labels.resize(nLabels);

    for (uint32_t i = 0; i < nLabels; ++i) {
        ValueType val = 0;
        file.read((char*)&val, 1);
        labels[i] = val;
    }

    return labels;
}
