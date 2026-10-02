package com.example.chebyshev.validation;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevValidationTest {

    @Test
    void testAdaptiveTruncate() {

        double[] coefficients = {
                1.0,
                0.5,
                1e-12,
                1e-14
        };

        double[] result =
                ChebyshevValidation.adaptiveTruncate(
                        coefficients,
                        1e-10
                );

        assertArrayEquals(
                new double[]{
                        1.0,
                        0.5
                },
                result,
                1e-12
        );
    }

    @Test
    void testAdaptiveTruncateKeepsSignificantCoefficient() {

        double[] coefficients = {
                1.0,
                0.5,
                0.01
        };

        double[] result =
                ChebyshevValidation.adaptiveTruncate(
                        coefficients,
                        1e-3
                );

        assertArrayEquals(
                coefficients,
                result,
                1e-12
        );
    }

    @Test
    void testResidual() {

        /*
         * f = 6
         * g = 2
         * q = 3
         *
         * R = 6 - 2*3 = 0
         */

        double[] f = {6.0};
        double[] g = {2.0};
        double[] q = {3.0};

        double result =
                ChebyshevValidation.residual(
                        f,
                        g,
                        q,
                        0.5
                );

        assertEquals(
                0.0,
                result,
                1e-12
        );
    }

    @Test
    void testNonZeroResidual() {

        /*
         * f = 6
         * g = 2
         * q = 2
         *
         * R = 6 - 4 = 2
         */

        double result =
                ChebyshevValidation.residual(
                        new double[]{6.0},
                        new double[]{2.0},
                        new double[]{2.0},
                        0.0
                );

        assertEquals(
                2.0,
                result,
                1e-12
        );
    }

    @Test
    void testMaxResidual() {

        double[] f = {
                6.0
        };

        double[] g = {
                2.0
        };

        double[] q = {
                3.0
        };

        double[] points = {
                -1.0,
                -0.5,
                0.0,
                0.5,
                1.0
        };

        double result =
                ChebyshevValidation.maxResidual(
                        f,
                        g,
                        q,
                        points
                );

        assertEquals(
                0.0,
                result,
                1e-12
        );
    }

    @Test
    void testSamplePoints() {

        double[] points =
                ChebyshevValidation.samplePoints(
                        0.0,
                        10.0,
                        5
                );

        assertArrayEquals(
                new double[]{
                        0.0,
                        2.5,
                        5.0,
                        7.5,
                        10.0
                },
                points,
                1e-12
        );
    }

    @Test
    void testHasPossibleRoot() {

        /*
         * g(x) = x
         *
         * In Chebyshev form:
         *
         * g(x) = T1(x)
         *
         * Root exists at x = 0.
         */

        double[] g = {
                0.0,
                1.0
        };

        assertTrue(
                ChebyshevValidation.hasPossibleRoot(
                        g,
                        -1.0,
                        1.0,
                        101
                )
        );
    }

    @Test
    void testNoRoot() {

        /*
         * g(x) = 2
         *
         * No root exists.
         */

        double[] g = {
                2.0
        };

        assertFalse(
                ChebyshevValidation.hasPossibleRoot(
                        g,
                        -1.0,
                        1.0,
                        100
                )
        );
    }

    @Test
    void testValidDivision() {

        /*
         * f = 6
         * g = 2
         * q = 3
         */

        boolean result =
                ChebyshevValidation.isValidDivision(
                        new double[]{6.0},
                        new double[]{2.0},
                        new double[]{3.0},
                        0.0,
                        10.0,
                        100,
                        1e-10
                );

        assertTrue(result);
    }

    @Test
    void testInvalidDivisionBecauseOfResidual() {

        /*
         * f = 6
         * g = 2
         * q = 2
         *
         * Residual = 2
         */

        boolean result =
                ChebyshevValidation.isValidDivision(
                        new double[]{6.0},
                        new double[]{2.0},
                        new double[]{2.0},
                        0.0,
                        10.0,
                        100,
                        1e-10
                );

        assertFalse(result);
    }

    @Test
    void testInvalidDivisionBecauseOfRoot() {

        /*
         * g(x) = x
         *
         * It has a root inside [-1,1].
         */

        boolean result =
                ChebyshevValidation.isValidDivision(
                        new double[]{1.0},
                        new double[]{0.0, 1.0},
                        new double[]{1.0},
                        -1.0,
                        1.0,
                        101,
                        1e-10
                );

        assertFalse(result);
    }

    @Test
    void testDivisionError() {

        double result =
                ChebyshevValidation.divisionError(
                        new double[]{6.0},
                        new double[]{2.0},
                        new double[]{2.0},
                        0.0,
                        10.0,
                        50
                );

        assertEquals(
                2.0,
                result,
                1e-12
        );
    }

    @Test
    void testInvalidTolerance() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevValidation.adaptiveTruncate(
                                new double[]{1.0},
                                0.0
                        )
        );
    }

    @Test
    void testInvalidSamples() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        ChebyshevValidation.samplePoints(
                                0.0,
                                1.0,
                                1
                        )
        );
    }
}