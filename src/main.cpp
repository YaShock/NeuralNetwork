#include "matrix.h"
#include "mnist_parser.h"
#include "network.h"
#include <algorithm>
#include <cassert>
#include <exception>
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
        // 2, 9, 7
        // 13, 15, 13
        Matrix<int> cv(2, 3, {2, 9, 7, 13, 15, 13});
        assert(a == cv);
        assert(a.size() == 6);
        Matrix<int> col(2, 1, {9, 15});
        Matrix<int> row(1, 3, {13, 15, 13});

        assert(a.row(1) == row);
        assert(a.col(1) == col);
        Matrix<int> cv2(2, 3, {4, 11, 9, 15, 17, 15});
        a += 2;
        assert(a == cv2);
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
    // Tranpose
    {
        // 1, 2, 4
        // 7, 8, 9
        Matrix<int> a(2, 3, {1, 2, 4, 7, 8, 9});
        // 1, 7
        // 2, 8
        // 4, 9
        Matrix<int> b{3, 2, {1, 7, 2, 8, 4, 9}};
        assert(a.transpose() == b);
        assert(b.transpose() == a);
    }
    // addColVec
    {
        // 1, 7
        // 2, 8
        // 4, 9
        Matrix<int> a{3, 2, {1, 7, 2, 8, 4, 9}};
        ColVec<int> b({3, 2, 1});
        a.addColVec(b);
        Matrix<int> c{3, 2, {4, 10, 4, 10, 5, 10}};
        assert(a == c);
    }
}

#define STR(a) ((std::ostringstream << a).str())

int main(int argc, char const* argv[])
{
    test_matrix();

    auto images = parseMnistImages("data/train-images.idx3-ubyte");
    auto labels = parseMnistLabels("data/train-labels.idx1-ubyte");
    auto testImages = parseMnistImages("data/t10k-images.idx3-ubyte");
    auto testLabels = parseMnistLabels("data/t10k-labels.idx1-ubyte");

    // std::cout << "Images size: " << images.rows() << "X" << images.cols() << std::endl;
    // std::cout << "Labels size: " << labels.rows() << "X" << labels.cols() << std::endl;

    // std::cout << "First image\n";
    // Matrix<double> firstCol = images.col(0);
    // Matrix<char> image(28, 28);
    // int counter = 0;
    // for (int i = 0; i < 28; ++i) {
    //     for (int j = 0; j < 28; ++j) {
    //         image(i, j) = static_cast<char>(firstCol(counter, 0));
    //         ++counter;
    //     }
    // }
    // std::cout << image << std::endl;
    // std::cout << "First label\n";
    // std::cout << labels.col(0) << std::endl;

    // Dummy  test set
    Matrix<double> dummyData(1, 100000, 0);
    Matrix<double> dummyLabels(1, 100000, 0);

    std::normal_distribution<double> distribution(-3.14, 3.14);
    std::mt19937 numGenerator;
    for (int i = 0; i < 100000; ++i) {
        double x = distribution(numGenerator);
        dummyData(0, i) = x;
        dummyLabels(0, i) = std::sin(x);
    }

    // std::vector<size_t> indeces(images.cols());
    // std::iota(indeces.begin(), indeces.end(), 0);
    // std::random_shuffle(indeces.begin(), indeces.end());

    // Matrix<double> shuffledImages(images.rows(), images.cols());
    // Matrix<double> shuffledLabels(labels.rows(), labels.cols());

    // for (size_t i = 0; i < images.cols(); ++i) {
    //     size_t newIdx = indeces[i];
    //     for (size_t j = 0; j < images.rows(); ++j) {
    //         shuffledImages(j, i) = images(j, newIdx);
    //     }
    //     for (size_t j = 0; j < labels.rows(); ++j) {
    //         shuffledLabels(j, i) = labels(j, newIdx);
    //     }
    // }

    // std::vector<int> counts(10);
    // for (size_t s = 0; s < labels.cols(); ++s) {
    //     for (size_t i = 0; i < labels.rows(); ++i) {
    //         if (labels(i, s) == 1) {
    //             ++counts[i];
    //             break;
    //         }
    //     }
    // }
    // for (size_t i = 0; i < 10; ++i) {
    //     std::cout << "Count " << i << ": " << counts[i] << std::endl;
    // }

    try {
        Network network({784, 30, 10});
        network.train(images, labels, 10, 30, 3, testImages, testLabels);
        // Network network({1, 10, 5, 5, 1});
        // network.train(dummyData, dummyLabels, 30, 1, 0.1, {}, {});

        // std::vector<Matrix<double>> weights = {
        //     Matrix<double>(3, 2, {0.8, 0.2, 0.4, 0.9, 0.3, 0.5}),
        //     Matrix<double>(1, 3, {0.3, 0.5, 0.9})};
        // std::vector<ColVec<double>> biases = {
        //     ColVec<double>({0, 0, 0}),
        //     ColVec<double>({0})};

        // input | output
        // --------------
        // 0, 0  | 0
        // 0, 1  | 1
        // 1, 0  | 1
        // 1, 1  | 0
        // Matrix<double> dummyData2(2, 4, {1, 1, 0, 1, 1, 0, 0, 0});
        // Matrix<double> dummyLabels2(1, 4, {0, 1, 1, 0});

        // Matrix<double> dummyData2(2, 4, {1, 1, 1, 1, 1, 1, 1, 1});
        // Matrix<double> dummyLabels2(1, 4, {0, 0, 0, 0});

        // Network network(weights, biases);
        // network.train(dummyData2, dummyLabels2, 1, 1, 1, {}, {});
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
