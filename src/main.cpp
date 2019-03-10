#include "matrix.h"
#include "mnist_parser.h"
#include "network.h"
#include <cassert>
#include <iostream>

void test_matrix()
{
    //
    {
        Matrix<int> a(2, 3, {1, 7, 3, 6, 7, 4});
        Matrix<int> b(3, 4, {1, 2, 4, 7, 8, 9, 6, 5, 3, 4, 7, 8});
        auto c = a * b;
        Matrix<int> cv(2, 4, {66, 77, 67, 66, 74, 91, 94, 109});
        assert(c == cv);
    }
    //
    {
        Matrix<int> a(2, 3, {1, 7, 3, 6, 7, 4});
        Matrix<int> b(2, 3, {1, 2, 4, 7, 8, 9});
        auto c = a + b;
        Matrix<int> cv(2, 3, {2, 9, 7, 13, 15, 13});
        assert(c == cv);
    }
    //
    {
        Matrix<int> a(2, 3, {1, 7, 3, 6, 7, 4});
        Matrix<int> b(2, 3, {1, 2, 4, 7, 8, 9});
        a += b;
        Matrix<int> cv(2, 3, {2, 9, 7, 13, 15, 13});
        assert(a == cv);
    }
    //
    {
        Matrix<int> a(1, 3, {1, 7, 3});
        Matrix<int> b(1, 3, {1, 2, 4});
        auto c = a.dot(b);
        assert(c == 27);
    }
    //
    {
        Matrix<int> a(1, 3, {1, 7, 3});
        Matrix<int> b(1, 3, {1, 2, 4});
        auto c = a.hamadard(b);
        Matrix<int> cv(1, 3, {1, 14, 12});
        assert(c == cv);
    }
}

int main(int argc, char const* argv[])
{
    test_matrix();

    auto images = parseMnistImages("data/t10k-images.idx3-ubyte");
    auto labels = parseMnistLabels("data/t10k-labels.idx1-ubyte");

    std::cout << "Images size: " << images.rows() << "X" << images.cols() << std::endl;
    std::cout << "Labels size: " << labels.rows() << "X" << labels.cols() << std::endl;

    std::cout << "First image\n";
    Matrix<double> firstCol = images.col(0);
    Matrix<char> image(28, 28);
    int counter = 0;
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            image(i, j) = static_cast<char>(firstCol(counter, 0));
            ++counter;
        }
    }
    std::cout << image << std::endl;
    std::cout << "First label\n";
    std::cout << labels.col(0) << std::endl;

    Network network({784, 30, 10});
    network.train(images, labels, 10, 3, 0.01);

    return 0;
}
