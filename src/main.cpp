#include "matrix.h"
#include "mnist_parser.h"
#include "network.h"
#include <algorithm>
#include <exception>
#include <iostream>

int main(int argc, char const* argv[])
{
    auto images = parseMnistImages("data/train-images.idx3-ubyte");
    auto labels = parseMnistLabels("data/train-labels.idx1-ubyte");
    auto testImages = parseMnistImages("data/t10k-images.idx3-ubyte");
    auto testLabels = parseMnistLabels("data/t10k-labels.idx1-ubyte");

    try {
        Network network({784, 30, 10, 10});
        network.train(images, labels, 10, 10, 3, testImages, testLabels);

        // for (size_t i = 0; i < 10; ++i) {
        //     auto firstLabel = testLabels.col(i);
        //     std::cout << "Expected: " << firstLabel.transpose() << std::endl;
        //     auto res = network.predict(testImages.col(i));
        //     std::cout << "Result: " << res.transpose() << std::endl;
        // }
    } catch (const std::exception& e) {
        std::cerr << e.what() << '\n';
    }

    return 0;
}
