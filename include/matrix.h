#ifndef MATRIX_H
#define MATRIX_H

#include <exception>
#include <iostream>
#include <iterator>
#include <random>
#include <vector>

template <typename Type>
class RowVec;

template <typename Type>
class ColVec;

// Row major matrix
template <typename T>
class Matrix
{
public:
    Matrix() : mRows(0), mCols(0)
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
    Matrix(size_t rows, size_t cols, const std::vector<T>& values) : mRows(rows), mCols(cols)
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

    ColVec<T> col(size_t c) const
    {
        ColVec<T> res(mRows);
        for (size_t r = 0; r < mRows; ++r) {
            res[r] = mData[mCols * r + c];
        }
        return res;
    }

    RowVec<T> row(size_t r) const
    {
        RowVec<T> res(mCols);
        for (size_t c = 0; c < mCols; ++c) {
            res[c] = mData[mCols * r + c];
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

    T& operator[](size_t idx)
    {
        return mData[idx];
    }

    const T& operator[](size_t idx) const
    {
        return mData[idx];
    }

    Matrix<T>& fill(T value)
    {
        for (T& x : mData) {
            x = value;
        }
        return *this;
    }

    template <typename Func>
    Matrix<T>& apply(Func func)
    {
        for (T& x : mData) {
            x = func(x);
        }
        return *this;
    }

    template <typename Func>
    Matrix<T> transform(Func func) const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = func(mData[mCols * i + j]);
            }
        }
        return res;
    }

    template <typename Pred>
    bool any(Pred pred) const
    {
        for (T x : mData) {
            if (pred(x)) {
                return true;
            }
        }
        return false;
    }

    template <typename Pred>
    bool all(Pred pred) const
    {
        for (T x : mData) {
            if (!pred(x)) {
                return false;
            }
        }
        return true;
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

    Matrix<T> operator+(T scalar) const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = mData[mCols * i + j] + scalar;
            }
        }
        return res;
    }

    Matrix<T>& operator+=(T scalar)
    {
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
                res(i, j) = mData[mCols * i + j] * scalar;
            }
        }
        return res;
    }

    Matrix<T>& operator*=(T scalar)
    {
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] *= scalar;
            }
        }
        return *this;
    }

    Matrix<T> operator/(T scalar) const
    {
        Matrix<T> res(mRows, mCols);
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                res(i, j) = mData[mCols * i + j] / scalar;
            }
        }
        return res;
    }

    Matrix<T>& operator/=(T scalar)
    {
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] /= scalar;
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

    Matrix<T>& addColVec(const ColVec<T>& vec)
    {
        if (mRows != vec.mRows) {
            throw std::invalid_argument("Matrix addColVec bad matrix sizes");
        }
        for (size_t i = 0; i < mRows; ++i) {
            for (size_t j = 0; j < mCols; ++j) {
                mData[mCols * i + j] += vec(i, 0);
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

    Matrix<T> operator*(const Matrix<T>& other) const
    {
        if (mCols != other.mRows) {
            throw std::invalid_argument("Matrix operator* bad matrix sizes");
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
            throw std::invalid_argument("Matrix operator* bad matrix sizes");
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

    Matrix<T> hamadard(const Matrix<T>& other) const
    {
        if (mRows != other.mRows || mCols != other.mCols) {
            throw std::invalid_argument("Matrix hamadard bad matrix sizes");
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
                if (mData[mCols * i + j] != other.mData[mCols * i + j]) {
                    return false;
                }
            }
        }
        return true;
    }

    T dot(const Matrix<T>& other)
    {
        if (!(mRows == 1 && other.mRows == 1) &&
            !(mCols == 1 && other.mCols == 1)) {
            throw std::invalid_argument("dot: matrixes are not vectors");
        }
        if (size() != other.size()) {
            throw std::invalid_argument("dot: vectors are not of equal size");
        }
        T res = T{};
        for (size_t i = 0; i < size(); ++i) {
            res += mData[i] * other.mData[i];
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
            if (i < matrix.rows() - 1) {
                os << '\n';
            }
        }
        return os;
    }

private:
    size_t mRows;
    size_t mCols;
    std::vector<T> mData;
};

template <typename T>
class RowVec : public Matrix<T>
{
public:
    RowVec() : Matrix<T>() {}
    RowVec(size_t cols) : Matrix<T>(1, cols) {}
    RowVec(size_t cols, T value) : Matrix<T>(1, cols, value) {}
    RowVec(const std::vector<T>& values) : Matrix<T>(1, values.size(), values) {}
    RowVec(const std::initializer_list<T>& values) : Matrix<T>(1, values.size(), values) {}
    RowVec(const Matrix<T>& matrix) : Matrix<T>(1, matrix.cols())
    {
        if (matrix.rows() != 1) {
            throw std::invalid_argument("RowVec ctor bad matrix sizes");
        }
        for (size_t i = 0; i < matrix.cols(); ++i) {
            (*this)[i] = matrix[i];
        }
    }
};

template <typename T>
class ColVec : public Matrix<T>
{
public:
    ColVec() : Matrix<T>() {}
    ColVec(size_t rows) : Matrix<T>(rows, 1) {}
    ColVec(size_t rows, T value) : Matrix<T>(rows, 1, value) {}
    ColVec(const std::vector<T>& values) : Matrix<T>(values.size(), 1, values) {}
    ColVec(const std::initializer_list<T>& values) : Matrix<T>(values.size(), 1, values) {}
    ColVec(const Matrix<T>& matrix) : Matrix<T>(matrix.rows(), 1)
    {
        if (matrix.cols() != 1) {
            throw std::invalid_argument("ColVec ctor bad matrix sizes");
        }
        for (size_t i = 0; i < matrix.rows(); ++i) {
            (*this)[i] = matrix[i];
        }
    }
};

#endif
