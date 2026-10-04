package com.example.chebyshev.solver;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevMatrixSolverTest {

    @Test
    void testSolveSimpleSystem() {

        /*
         * 2x + y = 5
         * x + 3y = 6
         *
         * Solution:
         *
         * x = 9/5 = 1.8
         * y = 7/5 = 1.4
         */

        double[][] A = {
                {2.0, 1.0},
                {1.0, 3.0}
        };

        double[] b = {
                5.0,
                6.0
        };

        double[] result =
                ChebyshevMatrixSolver.solve(
                        A,
                        b
                );

        assertEquals(
                1.8,
                result[0],
                1e-12
        );

        assertEquals(
                1.4,
                result[1],
                1e-12
        );
    }

    @Test
    void testSolveThreeByThreeSystem() {

        /*
         * x + y + z = 6
         * 2x + 3y + z = 10
         * x + 2y + 3z = 13
         *
         * Solution:
         *
         * x = 2
         * y = 1
         * z = 3
         */

        double[][] A = {
                {1.0, 1.0, 1.0},
                {2.0, 3.0, 1.0},
                {1.0, 2.0, 3.0}
        };

        double[] b = {
                6.0,
                10.0,
                13.0
        };

        double[] result =
                ChebyshevMatrixSolver.solve(
                        A,
                        b
                );

        assertEquals(
                2.0,
                result[0],
                1e-12
        );

        assertEquals(
                1.0,
                result[1],
                1e-12
        );

        assertEquals(
                3.0,
                result[2],
                1e-12
        );
    }

    @Test
    void testPartialPivoting() {

        /*
         * The first pivot is zero,
         * therefore row swapping is required.
         */

        double[][] A = {
                {0.0, 2.0},
                {1.0, 3.0}
        };

        double[] b = {
                4.0,
                7.0
        };

        double[] result =
                ChebyshevMatrixSolver.solve(
                        A,
                        b
                );

        /*
         * 2y = 4 => y = 2
         * x + 3(2) = 7 => x = 1
         */

        assertEquals(
                1.0,
                result[0],
                1e-12
        );

        assertEquals(
                2.0,
                result[1],
                1e-12
        );
    }

    @Test
    void testBuildSystemMatrixForConstant() {

        /*
         * g(x) = 2
         *
         * Multiplication by g should give:
         *
         * 2 * T_j = 2T_j
         *
         * Therefore the matrix is:
         *
         * [2 0 0]
         * [0 2 0]
         * [0 0 2]
         */

        double[] g = {
                2.0
        };

        double[][] matrix =
                ChebyshevMatrixSolver.buildSystemMatrix(
                        g,
                        2
                );

        assertEquals(
                2.0,
                matrix[0][0],
                1e-12
        );

        assertEquals(
                2.0,
                matrix[1][1],
                1e-12
        );

        assertEquals(
                2.0,
                matrix[2][2],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[0][1],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[1][0],
                1e-12
        );
    }

    @Test
    void testBuildSystemMatrixForT1() {

        /*
         * g(x) = T1
         *
         * T1*T0 = T1
         *
         * T1*T1 = 1/2(T2 + T0)
         *
         * T1*T2 = 1/2(T3 + T1)
         *
         * With degree 2 truncation:
         *
         *       T0   T1   T2
         *
         * T0    0   1/2   0
         * T1    1   0    1/2
         * T2    0  1/2    0
         */

        double[] g = {
                0.0,
                1.0
        };

        double[][] matrix =
                ChebyshevMatrixSolver.buildSystemMatrix(
                        g,
                        2
                );

        assertEquals(
                0.0,
                matrix[0][0],
                1e-12
        );

        assertEquals(
                0.5,
                matrix[0][1],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[0][2],
                1e-12
        );

        assertEquals(
                1.0,
                matrix[1][0],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[1][1],
                1e-12
        );

        assertEquals(
                0.5,
                matrix[1][2],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[2][0],
                1e-12
        );

        assertEquals(
                0.5,
                matrix[2][1],
                1e-12
        );

        assertEquals(
                0.0,
                matrix[2][2],
                1e-12
        );
    }

    @Test
    void testDivideByConstant() {

        /*
         * f(x) = 6
         * g(x) = 2
         *
         * q(x) = 3
         */

        double[] f = {
                6.0
        };

        double[] g = {
                2.0
        };

        double[] q =
                ChebyshevMatrixSolver.divide(
                        f,
                        g,
                        0
                );

        assertEquals(
                3.0,
                q[0],
                1e-12
        );
    }

    @Test
    void testDivideSimpleSeries() {

        /*
         * f(x) = 2 + 2T1
         * g(x) = 2
         *
         * q(x) = 1 + T1
         */

        double[] f = {
                2.0,
                2.0
        };

        double[] g = {
                2.0
        };

        double[] q =
                ChebyshevMatrixSolver.divide(
                        f,
                        g,
                        1
                );

        assertEquals(
                1.0,
                q[0],
                1e-12
        );

        assertEquals(
                1.0,
                q[1],
                1e-12
        );
    }

    @Test
    void testEvaluate() {

        /*
         * f(x) = 1 + 2T1
         */

        double[] coefficients = {
                1.0,
                2.0
        };

        double actual =
                ChebyshevMatrixSolver.evaluate(
                        coefficients,
                        0.5
                );

        assertEquals(
                2.0,
                actual,
                1e-12
        );
    }

    @Test
    void testSingularMatrix() {

        double[][] A = {
                {1.0, 2.0},
                {2.0, 4.0}
        };

        double[] b = {
                3.0,
                6.0
        };

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevMatrixSolver.solve(
                                A,
                                b
                        )
        );
    }

    @Test
    void testNonSquareMatrix() {

        double[][] A = {
                {1.0, 2.0, 3.0},
                {4.0, 5.0, 6.0}
        };

        double[] b = {
                1.0,
                2.0
        };

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevMatrixSolver.solve(
                                A,
                                b
                        )
        );
    }

    @Test
    void testInvalidVectorSize() {

        double[][] A = {
                {1.0, 0.0},
                {0.0, 1.0}
        };

        double[] b = {
                1.0
        };

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevMatrixSolver.solve(
                                A,
                                b
                        )
        );
    }

    @Test
    void testInvalidDegree() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevMatrixSolver.buildSystemMatrix(
                                new double[]{1.0},
                                -1
                        )
        );
    }

    @Test
    void testBuildToeplitzPart() {

        double[] g = {
                2.0,
                3.0,
                4.0
        };

        double[][] matrix =
                ChebyshevMatrixSolver.buildToeplitzPart(
                        g,
                        2
                );

        assertArrayEquals(
                new double[]{2.0, 3.0, 4.0},
                matrix[0],
                1e-12
        );

        assertArrayEquals(
                new double[]{3.0, 2.0, 3.0},
                matrix[1],
                1e-12
        );

        assertArrayEquals(
                new double[]{4.0, 3.0, 2.0},
                matrix[2],
                1e-12
        );
    }

    @Test
    void testBuildHankelPart() {

        double[] g = {
                2.0,
                3.0,
                4.0
        };

        double[][] matrix =
                ChebyshevMatrixSolver.buildHankelPart(
                        g,
                        2
                );

        assertArrayEquals(
                new double[]{2.0, 3.0, 4.0},
                matrix[0],
                1e-12
        );

        assertArrayEquals(
                new double[]{3.0, 4.0, 0.0},
                matrix[1],
                1e-12
        );

        assertArrayEquals(
                new double[]{4.0, 0.0, 0.0},
                matrix[2],
                1e-12
        );
    }

    @Test
    void testBuildToeplitzHankelMatrix() {

        double[] g = {
                2.0,
                3.0,
                4.0
        };

        double[][] matrix =
                ChebyshevMatrixSolver.buildToeplitzHankelMatrix(
                        g,
                        2
                );

        /*
         * A = 1/2 (T + H)
         *
         * Expected:
         *
         * [2   3   4/2]
         * [3   2   3/2]
         * [4/2 3/2 1]
         */

        assertEquals(
                2.0,
                matrix[0][0],
                1e-12
        );

        assertEquals(
                3.0,
                matrix[0][1],
                1e-12
        );

        assertEquals(
                2.0,
                matrix[0][2],
                1e-12
        );

        assertEquals(
                3.0,
                matrix[1][0],
                1e-12
        );

        assertEquals(
                2.0,
                matrix[1][1],
                1e-12
        );

        assertEquals(
                1.5,
                matrix[1][2],
                1e-12
        );

        assertEquals(
                2.0,
                matrix[2][0],
                1e-12
        );

        assertEquals(
                1.5,
                matrix[2][1],
                1e-12
        );

        assertEquals(
                1.0,
                matrix[2][2],
                1e-12
        );
    }

    @Test
    void testSolveToeplitzHankel() {

        /*
         * g(x) = 2
         *
         * Therefore:
         *
         * g(x) * q(x) = f(x)
         *
         * with
         *
         * f(x) = 6 + 4T1 + 2T2
         *
         * gives
         *
         * q(x) = 3 + 2T1 + T2
         */

        double[] g = {
                2.0
        };

        double[] f = {
                6.0,
                4.0,
                2.0
        };

        double[] q =
                ChebyshevMatrixSolver.solveToeplitzHankel(
                        g,
                        f,
                        2
                );

        assertEquals(
                3.0,
                q[0],
                1e-12
        );

        assertEquals(
                2.0,
                q[1],
                1e-12
        );

        assertEquals(
                1.0,
                q[2],
                1e-12
        );
    }


}