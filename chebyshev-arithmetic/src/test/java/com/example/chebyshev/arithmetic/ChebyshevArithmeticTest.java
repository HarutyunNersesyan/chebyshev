package com.example.chebyshev.arithmetic;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevArithmeticTest {

    @Test
    void testAdd() {

        double[] a = {
                1.0,
                2.0,
                3.0
        };

        double[] b = {
                4.0,
                5.0,
                6.0
        };

        double[] result =
                ChebyshevArithmetic.add(a, b);

        assertArrayEquals(
                new double[]{
                        5.0,
                        7.0,
                        9.0
                },
                result,
                1e-12
        );
    }

    @Test
    void testAddDifferentLengths() {

        double[] a = {
                1.0,
                2.0
        };

        double[] b = {
                3.0,
                4.0,
                5.0
        };

        double[] result =
                ChebyshevArithmetic.add(a, b);

        assertArrayEquals(
                new double[]{
                        4.0,
                        6.0,
                        5.0
                },
                result,
                1e-12
        );
    }

    @Test
    void testSubtract() {

        double[] a = {
                5.0,
                7.0,
                9.0
        };

        double[] b = {
                1.0,
                2.0,
                3.0
        };

        double[] result =
                ChebyshevArithmetic.subtract(a, b);

        assertArrayEquals(
                new double[]{
                        4.0,
                        5.0,
                        6.0
                },
                result,
                1e-12
        );
    }

    @Test
    void testMultiplyByConstant() {

        /*
         * 2 * (1 + 3T1)
         *
         * = 2 + 6T1
         */

        double[] a = {
                2.0
        };

        double[] b = {
                1.0,
                3.0
        };

        double[] result =
                ChebyshevArithmetic.multiply(a, b);

        assertArrayEquals(
                new double[]{
                        2.0,
                        6.0
                },
                result,
                1e-12
        );
    }

    @Test
    void testMultiplyT1T1() {

        /*
         * T1 * T1
         *
         * = x²
         *
         * = 1/2(T2 + T0)
         */

        double[] t1 = {
                0.0,
                1.0
        };

        double[] result =
                ChebyshevArithmetic.multiply(
                        t1,
                        t1
                );

        assertArrayEquals(
                new double[]{
                        0.5,
                        0.0,
                        0.5
                },
                result,
                1e-12
        );
    }

    @Test
    void testChebyshevMultiplicationIdentity() {

        /*
         * T2 * T3
         *
         * = 1/2(T5 + T1)
         */

        double[] t2 = {
                0.0,
                0.0,
                1.0
        };

        double[] t3 = {
                0.0,
                0.0,
                0.0,
                1.0
        };

        double[] result =
                ChebyshevArithmetic.multiply(
                        t2,
                        t3
                );

        assertArrayEquals(
                new double[]{
                        0.0,
                        0.5,
                        0.0,
                        0.0,
                        0.0,
                        0.5
                },
                result,
                1e-12
        );
    }

    @Test
    void testEvaluate() {

        /*
         * f(x) = 1 + 2T1 + 3T2
         */

        double[] coefficients = {
                1.0,
                2.0,
                3.0
        };

        double x = 0.25;

        double expected =
                1.0
                        + 2.0 * x
                        + 3.0 * (2.0 * x * x - 1.0);

        double actual =
                ChebyshevArithmetic.evaluate(
                        coefficients,
                        x
                );

        assertEquals(
                expected,
                actual,
                1e-12
        );
    }

    @Test
    void testInverseConstant() {

        /*
         * g(x) = 2
         *
         * 1/g = 0.5
         */

        double[] g = {
                2.0
        };

        double[] inverse =
                ChebyshevArithmetic.inverse(
                        g,
                        5
                );

        assertEquals(
                0.5,
                inverse[0],
                1e-12
        );
    }

    @Test
    void testInverseLinearSeries() {

        /*
         * g(x) = 1 + 0.1 T1(x)
         *
         * The inverse is represented by a truncated
         * Chebyshev series, therefore:
         *
         * g(x) * inverse(g)(x) ≈ 1
         */

        double[] g = {
                1.0,
                0.1
        };

        double[] inverse =
                ChebyshevArithmetic.inverse(
                        g,
                        10
                );

        double[] product =
                ChebyshevArithmetic.multiply(
                        g,
                        inverse
                );

        /*
         * Because the inverse is truncated to degree 1,
         * a small approximation error is expected.
         */

        assertEquals(
                1.0,
                ChebyshevArithmetic.evaluate(
                        product,
                        0.0
                ),
                0.01
        );

        assertEquals(
                1.0,
                ChebyshevArithmetic.evaluate(
                        product,
                        0.5
                ),
                0.01
        );

        assertEquals(
                1.0,
                ChebyshevArithmetic.evaluate(
                        product,
                        -0.5
                ),
                0.01
        );
    }

    @Test
    void testDivideConstant() {

        /*
         * 6 / 2 = 3
         */

        double[] f = {
                6.0
        };

        double[] g = {
                2.0
        };

        double[] result =
                ChebyshevArithmetic.divide(
                        f,
                        g,
                        5
                );

        assertEquals(
                3.0,
                result[0],
                1e-12
        );
    }

    @Test
    void testTToUConversion() {

        /*
         * T2 = 1/2(U2 - U0)
         */

        double[] t = {
                0.0,
                0.0,
                1.0
        };

        double[] u =
                ChebyshevArithmetic.toU(t);

        assertArrayEquals(
                new double[]{
                        -0.5,
                        0.0,
                        0.5
                },
                u,
                1e-12
        );
    }

    @Test
    void testUToTConversion() {

        /*
         * U2 = 2T2 + T0
         */

        double[] u = {
                0.0,
                0.0,
                1.0
        };

        double[] t =
                ChebyshevArithmetic.toT(u);

        assertArrayEquals(
                new double[]{
                        1.0,
                        0.0,
                        2.0
                },
                t,
                1e-12
        );
    }

    @Test
    void testRoundTripTToUToT() {

        double[] original = {
                1.0,
                2.0,
                3.0,
                4.0
        };

        double[] u =
                ChebyshevArithmetic.toU(
                        original
                );

        double[] restored =
                ChebyshevArithmetic.toT(
                        u
                );

        assertArrayEquals(
                original,
                restored,
                1e-12
        );
    }

    @Test
    void testInvalidCoefficients() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevArithmetic.add(
                                null,
                                new double[]{1.0}
                        )
        );

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevArithmetic.multiply(
                                new double[]{1.0},
                                null
                        )
        );
    }

    @Test
    void testInvalidIterations() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevArithmetic.inverse(
                                new double[]{1.0},
                                0
                        )
        );
    }

    @Test
    void testZeroConstantInverseFails() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevArithmetic.inverse(
                                new double[]{0.0, 1.0},
                                5
                        )
        );
    }
}