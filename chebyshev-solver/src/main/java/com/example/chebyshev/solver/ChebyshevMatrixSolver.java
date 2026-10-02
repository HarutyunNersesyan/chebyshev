package com.example.chebyshev.solver;

import com.example.chebyshev.core.ChebyshevCore;

public final class ChebyshevMatrixSolver {

    private ChebyshevMatrixSolver() {
        // Utility class
    }

    /**
     * Solves a linear system:
     *
     * A * x = b
     *
     * using Gaussian elimination with
     * partial pivoting.
     */
    public static double[] solve(
            double[][] A,
            double[] b
    ) {
        validateMatrix(A);
        validateVector(b, A.length);

        int n = A.length;

        double[][] matrix =
                copyMatrix(A);

        double[] rhs =
                b.clone();

        /*
         * Forward elimination
         */
        for (int column = 0;
             column < n;
             column++) {

            /*
             * Find pivot row.
             */
            int pivotRow = column;

            double maxValue =
                    Math.abs(matrix[column][column]);

            for (int row = column + 1;
                 row < n;
                 row++) {

                double value =
                        Math.abs(matrix[row][column]);

                if (value > maxValue) {
                    maxValue = value;
                    pivotRow = row;
                }
            }

            /*
             * Singular matrix check.
             */
            if (maxValue < 1e-14) {
                throw new IllegalArgumentException(
                        "Matrix is singular or nearly singular"
                );
            }

            /*
             * Swap rows.
             */
            if (pivotRow != column) {

                double[] tempRow =
                        matrix[column];

                matrix[column] =
                        matrix[pivotRow];

                matrix[pivotRow] =
                        tempRow;

                double temp =
                        rhs[column];

                rhs[column] =
                        rhs[pivotRow];

                rhs[pivotRow] =
                        temp;
            }

            /*
             * Eliminate values below pivot.
             */
            for (int row = column + 1;
                 row < n;
                 row++) {

                double factor =
                        matrix[row][column]
                                / matrix[column][column];

                matrix[row][column] = 0.0;

                for (int j = column + 1;
                     j < n;
                     j++) {

                    matrix[row][j] -=
                            factor * matrix[column][j];
                }

                rhs[row] -=
                        factor * rhs[column];
            }
        }

        /*
         * Back substitution.
         */
        double[] x =
                new double[n];

        for (int row = n - 1;
             row >= 0;
             row--) {

            double sum =
                    rhs[row];

            for (int j = row + 1;
                 j < n;
                 j++) {

                sum -=
                        matrix[row][j] * x[j];
            }

            x[row] =
                    sum / matrix[row][row];
        }

        return x;
    }

    /**
     * Builds a Chebyshev multiplication matrix.
     *
     * The matrix represents multiplication by
     * a Chebyshev series:
     *
     * g(x) = g_0 T_0 + ... + g_m T_m
     *
     * using the identity:
     *
     * T_m T_n =
     * 1/2 [T_{m+n} + T_{|m-n|}]
     *
     * The resulting matrix is truncated to
     * the requested degree.
     */
    public static double[][] buildSystemMatrix(
            double[] g,
            int degree
    ) {
        validateCoefficients(g);

        if (degree < 0) {
            throw new IllegalArgumentException(
                    "Degree must be non-negative"
            );
        }

        int size = degree + 1;

        double[][] matrix =
                new double[size][size];

        /*
         * Each column corresponds to:
         *
         * g(x) * T_j(x)
         */
        for (int j = 0;
             j < size;
             j++) {

            for (int k = 0;
                 k < g.length;
                 k++) {

                double coefficient =
                        g[k];

                if (Math.abs(coefficient) < 1e-15) {
                    continue;
                }

                /*
                 * T_k * T_j
                 *
                 * = 1/2(
                 * T_{k+j}
                 * +
                 * T_|k-j|
                 * )
                 */

                int sumIndex =
                        k + j;

                int diffIndex =
                        Math.abs(k - j);

                double value =
                        0.5 * coefficient;

                if (sumIndex < size) {
                    matrix[sumIndex][j] += value;
                }

                if (diffIndex < size) {
                    matrix[diffIndex][j] += value;
                }
            }
        }

        return matrix;
    }

    /**
     * Solves approximately:
     *
     * g(x) * q(x) = f(x)
     *
     * where f and g are Chebyshev coefficient arrays.
     *
     * The solution q is truncated to the
     * requested degree.
     */
    public static double[] divide(
            double[] f,
            double[] g,
            int degree
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        if (degree < 0) {
            throw new IllegalArgumentException(
                    "Degree must be non-negative"
            );
        }

        int size = degree + 1;

        double[] rhs =
                new double[size];

        for (int i = 0;
             i < Math.min(f.length, size);
             i++) {

            rhs[i] = f[i];
        }

        double[][] matrix =
                buildSystemMatrix(
                        g,
                        degree
                );

        return solve(
                matrix,
                rhs
        );
    }

    /**
     * Evaluates a Chebyshev coefficient array.
     */
    public static double evaluate(
            double[] coefficients,
            double x
    ) {
        validateCoefficients(coefficients);

        return ChebyshevCore.clenshawT(
                coefficients,
                x
        );
    }

    private static void validateMatrix(
            double[][] matrix
    ) {
        if (matrix == null
                || matrix.length == 0) {

            throw new IllegalArgumentException(
                    "Matrix must not be null or empty"
            );
        }

        int n =
                matrix.length;

        for (double[] row : matrix) {

            if (row == null
                    || row.length != n) {

                throw new IllegalArgumentException(
                        "Matrix must be square"
                );
            }

            for (double value : row) {

                if (!Double.isFinite(value)) {
                    throw new IllegalArgumentException(
                            "Matrix values must be finite"
                    );
                }
            }
        }
    }

    private static void validateVector(
            double[] vector,
            int expectedSize
    ) {
        if (vector == null
                || vector.length != expectedSize) {

            throw new IllegalArgumentException(
                    "Vector size must match matrix size"
            );
        }

        for (double value : vector) {

            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Vector values must be finite"
                );
            }
        }
    }

    private static void validateCoefficients(
            double[] coefficients
    ) {
        if (coefficients == null
                || coefficients.length == 0) {

            throw new IllegalArgumentException(
                    "Coefficients must not be null or empty"
            );
        }

        for (double coefficient : coefficients) {

            if (!Double.isFinite(coefficient)) {
                throw new IllegalArgumentException(
                        "Coefficients must be finite"
                );
            }
        }
    }

    private static double[][] copyMatrix(
            double[][] matrix
    ) {
        double[][] copy =
                new double[matrix.length][];

        for (int i = 0;
             i < matrix.length;
             i++) {

            copy[i] =
                    matrix[i].clone();
        }

        return copy;
    }
}