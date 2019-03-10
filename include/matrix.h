#ifndef MATRIX_H
#define MATRIX_H

#include <exception>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

// Row major matrix
template <typename T>
class Matrix
{
public:
    Matrix()
    {
    }
    Matrix(size_t rows, size_t cols) : mRows(rows), mCols(cols)
    {
        mData.resize(rows * cols);
    }
    Matrix(size_t rows, size_t cols, T value) : mRows(rows), mCols(cols)
    {
        mData.resize(rows * cols);
        std::fill(mData.begin(), mData.end(), value);
    }
    Matrix(size_t rows, size_t cols, std::initializer_list<T> values) : mRows(rows), mCols(cols)
    {
        if (values.size() != rows * cols) {
            throw std::invalid_argument("Matrix ctor number of values and matrix size mismatch");
        }
        mData.resize(rows * cols);
        std::copy(values.begin(), values.end(), mData.begin());
    }

    size_t rows() const
    {
        return mRows;
    }

    size_t cols() const
    {
        return mCols;
    }

    Matrix<T> col(size_t c) const
    {
        Matrix<T> res(mRows, 1);
        for (size_t r = 0; r < mRows; ++r) {
            res(r, 0) = mData[mCols * r + c];
        }
        return res;
    }

    Matrix<T> row(size_t r) const
    {
        Matrix<T> res(1, mCols);
        for (size_t c = 0; c < mCols; ++c) {
            res(0, c) = mData[mCols * r + c];
        }
        return res;
    }

    size_t size() const
    {
        return mRows * mCols;
    }

    T& operator()(size_t row, size_t col)
    {
        return mData[mCols * row + col];
    }

    const T& operator()(size_t row, size_t col) const
    {
        return mData[mCols * row + col];
    }

    template <typename Distribution, typename NumGen>
    void randomize(Distribution distribution, NumGen numGen)
    {
        for (T& x : mData) {
            x = distribution(numGen);
        }
    }

    Matrix<T> transpose() const
    {
        Matrix<T> res(mCols, mRows);
        for (size_t r = 0; r < mRows; ++r) {
            for (size_t c = 0; c < mCols; ++c) {
                res(c, r) = mData[mCols * r + c];
            }
        }
        return res;
    }

    Matrix<T> operator*(const Matrix<T>& other) const
    {
        if (mCols != other.mRows) {
            throw std::invalid_argument("Matrix operator+ bad matrix sizes");
        }
        Matrix<T> res(mRows, other.mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < other.mCols; ++j) {
                T sum = T{};
                for (size_t k = 0; k < mCols; ++k) {
                    sum += mData[mCols * i + k] * other(k, j);
                }
                res(i, j) = sum;
            }
        }
        return res;
    }

    Matrix<T>& operator*=(const Matrix<T>& other)
    {
        if (mCols != other.mRows) {
            throw std::invalid_argument("Matrix operator+ bad matrix sizes");
        }
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                T sum = T{};
                for (size_t k = 0; k < mCols; ++k) {
                    sum += mData[mCols * i + k] * other(k, j);
                }
                mData[mCols * i + j] = sum;
            }
        }
        return *this;
    }

    Matrix<T> operator+(T scalar) const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) += scalar;
            }
        }
        return res;
    }

    Matrix<T>& operator+=(T scalar)
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] += scalar;
            }
        }
        return *this;
    }

    Matrix<T> operator*(T scalar) const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) *= scalar;
            }
        }
        return res;
    }

    Matrix<T>& operator*=(T scalar)
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] *= scalar;
            }
        }
        return *this;
    }

    Matrix<T> operator+(const Matrix<T>& other) const
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix operator+ bad matrix sizes");
        }
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = mData[mCols * i + j] + other(i, j);
            }
        }
        return res;
    }

    Matrix<T>& operator+=(const Matrix<T>& other)
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix operator+= bad matrix sizes");
        }
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] += other(i, j);
            }
        }
        return *this;
    }

    Matrix<T> operator-(const Matrix<T>& other) const
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix operator- bad matrix sizes");
        }
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = mData[mCols * i + j] - other(i, j);
            }
        }
        return res;
    }

    Matrix<T>& operator-=(const Matrix<T>& other)
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix operator-= bad matrix sizes");
        }
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] -= other(i, j);
            }
        }
        return *this;
    }

    Matrix<T> hamadard(const Matrix<T>& other) const
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix operator== bad matrix sizes");
        }
        Matrix<T> res(mRows, mCols);

        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = mData[mCols * i + j] * other(i, j);
            }
        }

        return res;
    }

    bool operator==(const Matrix<T>& other)
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            return false;
        }
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                if (mData[mRows * i + j] != other.mData[mRows * i + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    T dot(const Matrix<T>& other)
    {
        if (mRows != 1 || other.mRows != 1) {
            throw std::invalid_argument("Matrixes are not vectors (Rows are not 1");
        }
        T res = T{};
        for (size_t i = 0; i < mCols; ++i) {
            res += mData[mRows * i] * other(0, i);
        }
        return res;
    }

    static Matrix<T> Identity(size_t rows, size_t cols)
    {
        Matrix<T> I(rows, cols, 0);
        for (size_t i = 0; i < std::min(rows, cols); ++i) {
            I(i, i) = 1;
        }
        return I;
    }

    friend std::ostream& operator<<(std::ostream& os, const Matrix<T>& matrix)
    {
        for (size_t i = 0; i < matrix.rows(); ++i) {
            for (size_t j = 0; j < matrix.cols(); ++j) {
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
    size_t mRows;
    size_t mCols;
    std::vector<T> mData;
};

#endif
