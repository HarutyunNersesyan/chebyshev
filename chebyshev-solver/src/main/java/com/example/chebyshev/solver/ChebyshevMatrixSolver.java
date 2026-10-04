package com.example.chebyshev.solver;

import com.example.chebyshev.core.ChebyshevCore;

public final class ChebyshevMatrixSolver {

    private static final double EPSILON = 1e-14;

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

        double[][] matrix = copyMatrix(A);
        double[] rhs = b.clone();

        /*
         * Forward elimination
         */
        for (int column = 0;
             column < n;
             column++) {

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

            if (maxValue < EPSILON) {
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
     * Builds the Toeplitz part of the Chebyshev
     * multiplication matrix.
     *
     * T(i,j) = g[|i-j|]
     */
    public static double[][] buildToeplitzPart(
            double[] g,
            int degree
    ) {
        validateCoefficients(g);
        validateDegree(degree);

        int size = degree + 1;

        double[][] matrix =
                new double[size][size];

        for (int i = 0; i < size; i++) {

            for (int j = 0; j < size; j++) {

                int index =
                        Math.abs(i - j);

                if (index < g.length) {
                    matrix[i][j] =
                            g[index];
                }
            }
        }

        return matrix;
    }

    /**
     * Builds the Hankel part of the Chebyshev
     * multiplication matrix.
     *
     * H(i,j) = g[i+j]
     */
    public static double[][] buildHankelPart(
            double[] g,
            int degree
    ) {
        validateCoefficients(g);
        validateDegree(degree);

        int size = degree + 1;

        double[][] matrix =
                new double[size][size];

        for (int i = 0; i < size; i++) {

            for (int j = 0; j < size; j++) {

                int index = i + j;

                if (index < g.length) {
                    matrix[i][j] =
                            g[index];
                }
            }
        }

        return matrix;
    }

    /**
     * Builds the Toeplitz + Hankel matrix
     * corresponding to Chebyshev multiplication.
     *
     * A = 1/2 (T + H)
     *
     * where
     *
     * T(i,j) = g[|i-j|]
     *
     * H(i,j) = g[i+j]
     *
     * Therefore:
     *
     * A(i,j) =
     * 1/2 * (g[|i-j|] + g[i+j])
     *
     * whenever the corresponding coefficient exists.
     */
    public static double[][] buildToeplitzHankelMatrix(
            double[] g,
            int degree
    ) {
        validateCoefficients(g);
        validateDegree(degree);

        int size = degree + 1;

        double[][] toeplitz =
                buildToeplitzPart(g, degree);

        double[][] hankel =
                buildHankelPart(g, degree);

        double[][] matrix =
                new double[size][size];

        for (int i = 0; i < size; i++) {

            for (int j = 0; j < size; j++) {

                matrix[i][j] =
                        0.5 * (
                                toeplitz[i][j]
                                        + hankel[i][j]
                        );
            }
        }

        return matrix;
    }

    /**
     * Solves a Toeplitz + Hankel Chebyshev
     * multiplication system:
     *
     * A * q = f
     *
     * where
     *
     * A = 1/2 (T + H)
     *
     * This method preserves the structured
     * Chebyshev representation while using
     * the existing stable Gaussian solver
     * with partial pivoting.
     */
    public static double[] solveToeplitzHankel(
            double[] g,
            double[] f,
            int degree
    ) {
        validateCoefficients(g);
        validateCoefficients(f);
        validateDegree(degree);

        int size = degree + 1;

        double[] rhs =
                new double[size];

        for (int i = 0;
             i < Math.min(f.length, size);
             i++) {

            rhs[i] = f[i];
        }

        double[][] matrix =
                buildToeplitzHankelMatrix(
                        g,
                        degree
                );

        return solve(
                matrix,
                rhs
        );
    }

    /**
     * Builds a Chebyshev multiplication matrix.
     *
     * The matrix represents multiplication by
     * a Chebyshev series:
     *
     * g(x) = g_0 T_0 + ... + g_m T_m
     *
     * using:
     *
     * T_m T_n =
     * 1/2 [T_{m+n} + T_|m-n|]
     *
     * The matrix is truncated to the
     * requested degree.
     */
    public static double[][] buildSystemMatrix(
            double[] g,
            int degree
    ) {
        return buildToeplitzHankelMatrix(
                g,
                degree
        );
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
        validateDegree(degree);

        return solveToeplitzHankel(
                g,
                f,
                degree
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

    private static void validateDegree(
            int degree
    ) {
        if (degree < 0) {
            throw new IllegalArgumentException(
                    "Degree must be non-negative"
            );
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