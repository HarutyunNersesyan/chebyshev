package com.example.chebyshev.core;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevCoreTest {

    @Test
    void testT0() {
        assertEquals(1.0, ChebyshevCore.T(0, 0.5), 1e-12);
    }

    @Test
    void testT1() {
        assertEquals(0.5, ChebyshevCore.T(1, 0.5), 1e-12);
    }

    @Test
    void testTRecurrence() {
        double x = 0.3;

        for (int n = 2; n <= 10; n++) {
            double expected =
                    2.0 * x * ChebyshevCore.T(n - 1, x)
                            - ChebyshevCore.T(n - 2, x);

            assertEquals(
                    expected,
                    ChebyshevCore.T(n, x),
                    1e-12
            );
        }
    }

    @Test
    void testURecurrence() {
        double x = 0.3;

        for (int n = 2; n <= 10; n++) {
            double expected =
                    2.0 * x * ChebyshevCore.U(n - 1, x)
                            - ChebyshevCore.U(n - 2, x);

            assertEquals(
                    expected,
                    ChebyshevCore.U(n, x),
                    1e-12
            );
        }
    }

    @Test
    void testTTrigonometricIdentity() {
        double x = 0.25;

        for (int n = 0; n <= 10; n++) {
            assertEquals(
                    ChebyshevCore.T(n, x),
                    ChebyshevCore.tTrig(n, x),
                    1e-12
            );
        }
    }

    @Test
    void testUTrigonometricIdentity() {
        double x = 0.25;

        for (int n = 0; n <= 10; n++) {
            assertEquals(
                    ChebyshevCore.U(n, x),
                    ChebyshevCore.uTrig(n, x),
                    1e-12
            );
        }
    }

    @Test
    void testUBoundaries() {
        assertEquals(6.0, ChebyshevCore.U(5, 1.0), 1e-12);
        assertEquals(-6.0, ChebyshevCore.U(5, -1.0), 1e-12);
    }

    @Test
    void testDerivative() {
        double x = 0.4;

        for (int n = 1; n <= 10; n++) {
            double expected =
                    n * ChebyshevCore.U(n - 1, x);

            assertEquals(
                    expected,
                    ChebyshevCore.derivativeT(n, x),
                    1e-12
            );
        }
    }

    @Test
    void testIntegralT0() {
        double x = 0.4;

        assertEquals(
                x,
                ChebyshevCore.integralT(0, x),
                1e-12
        );
    }

    @Test
    void testIntegralT1() {
        double x = 0.4;

        assertEquals(
                x * x / 2.0,
                ChebyshevCore.integralT(1, x),
                1e-12
        );
    }

    @Test
    void testDefiniteIntegral() {
        double result =
                ChebyshevCore.definiteIntegralT(
                        0,
                        0.0,
                        1.0
                );

        assertEquals(1.0, result, 1e-12);
    }

    @Test
    void testClenshawT() {
        double[] coefficients = {
                1.0,
                2.0,
                3.0
        };

        double x = 0.2;

        double expected =
                coefficients[0] * ChebyshevCore.T(0, x)
                        + coefficients[1] * ChebyshevCore.T(1, x)
                        + coefficients[2] * ChebyshevCore.T(2, x);

        assertEquals(
                expected,
                ChebyshevCore.clenshawT(coefficients, x),
                1e-12
        );
    }

    @Test
    void testClenshawU() {
        double[] coefficients = {
                1.0,
                2.0,
                3.0
        };

        double x = 0.2;

        double expected =
                coefficients[0] * ChebyshevCore.U(0, x)
                        + coefficients[1] * ChebyshevCore.U(1, x)
                        + coefficients[2] * ChebyshevCore.U(2, x);

        assertEquals(
                expected,
                ChebyshevCore.clenshawU(coefficients, x),
                1e-12
        );
    }

    @Test
    void testInvalidDegree() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevCore.T(-1, 0.5)
        );
    }

    @Test
    void testInvalidX() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevCore.T(2, 2.0)
        );
    }

    @Test
    void testInvalidCoefficients() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevCore.clenshawT(null, 0.5)
        );
    }
}