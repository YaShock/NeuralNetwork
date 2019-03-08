#ifndef MATRIX_H
#define MATRIX_H

#include <array>
#include <iostream>
#include <iterator>
#include <vector>

// Row major matrix
template <typename T, int Rows, int Cols>
class Matrix
{
public:
    Matrix()
    {
        data.resize(Rows * Cols);
    }
    Matrix(T value)
    {
        data.resize(Rows * Cols, value);
    }
    Matrix(std::array<T, Rows * Cols> values)
    {
        data.reserve(values.size());
        std::copy(values.begin(), values.end(), std::back_inserter(data));
    }

    int rows() const
    {
        return Rows;
    }

    int cols() const
    {
        return Cols;
    }

    T& operator()(int row, int col)
    {
        return data[Cols * row + col];
    }

    const T& operator()(int row, int col) const
    {
        return data[Cols * row + col];
    }

    template <int Cols2>
    Matrix<T, Rows, Cols2> operator*(const Matrix<T, Cols, Cols2>& other)
    {
        Matrix<T, Rows, Cols2> res;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols2; ++j) {
                T sum = T{};
                for (int k = 0; k < Cols; ++k) {
                    sum += data[Cols * i + k] * other(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
    }

    Matrix<T, Rows, Cols>& operator*=(const Matrix<T, Cols, Cols>& other)
    {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                T sum = T{};
                for (int k = 0; k < Cols; ++k) {
                    sum += data[Cols * i + k] * other(k, j);
                }
                data[Cols * i + j] = sum;
            }
        }
        return *this;
    }

    Matrix<T, Rows, Cols> operator+(const Matrix<T, Rows, Cols>& other)
    {
        Matrix<T, Rows, Cols> res;
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                res(i, j) = data[Cols * i + j] + other(i, j);
            }
        }
        return res;
    }

    Matrix<T, Rows, Cols>& operator+=(const Matrix<T, Rows, Cols>& other)
    {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                data[Cols * i + j] += other(i, j);
            }
        }
        return *this;
    }

    bool operator==(const Matrix<T, Rows, Cols>& other)
    {
        for (int i = 0; i < Rows; ++i) {
            for (int j = 0; j < Cols; ++j) {
                if (data[Rows * i + j] != other.data[Rows * i + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    static Matrix<T, Rows, Cols> Identity()
    {
        Matrix<T, Rows, Cols> I(0);
        for (int i = 0; i < std::min(Rows, Cols); ++i) {
            I(i, i) = 1;
        }
        return I;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T, Rows, Cols>& matrix)
    {
        for (int i = 0; i < matrix.rows(); ++i) {
            for (int j = 0; j < matrix.cols(); ++j) {
                os << matrix(i, j);
                if (j < matrix.cols() - 1) {
                    os << ' ';
                }
            }
            os << '\n';
        }
        return os;
    }

private:
    std::vector<T> data;
};

#endif