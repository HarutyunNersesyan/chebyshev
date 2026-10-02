package com.example.chebyshev.api;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevDivisionClientTest {

    @Test
    void testDefaultConstructor() {

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        assertEquals(
                10,
                client.getIterations()
        );

        assertEquals(
                1e-10,
                client.getTolerance(),
                1e-15
        );
    }

    @Test
    void testCustomConfiguration() {

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient(
                        20,
                        1e-8
                );

        assertEquals(
                20,
                client.getIterations()
        );

        assertEquals(
                1e-8,
                client.getTolerance(),
                1e-15
        );
    }

    @Test
    void testDivideConstants() {

        /*
         * 6 / 2 = 3
         */

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        double[] result =
                client.divide(
                        new double[]{6.0},
                        new double[]{2.0}
                );

        assertEquals(
                3.0,
                result[0],
                1e-12
        );
    }

    @Test
    void testDivideSimpleSeriesDirectly() {

        /*
         * f(x) = 2 + 2T1
         * g(x) = 2
         *
         * q(x) = 1 + T1
         */

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        double[] result =
                client.divideDirect(
                        new double[]{
                                2.0,
                                2.0
                        },
                        new double[]{
                                2.0
                        },
                        1
                );

        assertEquals(
                1.0,
                result[0],
                1e-12
        );

        assertEquals(
                1.0,
                result[1],
                1e-12
        );
    }

    @Test
    void testDivideAndValidate() {

        /*
         * f = 6
         * g = 2
         * q = 3
         */

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        ChebyshevDivisionClient.DivisionResult result =
                client.divideAndValidate(
                        new double[]{6.0},
                        new double[]{2.0},
                        100
                );

        assertTrue(
                result.isValid()
        );

        assertEquals(
                0.0,
                result.getResidual(),
                1e-12
        );

        assertEquals(
                3.0,
                result.getQuotient()[0],
                1e-12
        );
    }

    @Test
    void testDivideAndValidatePhysicalInterval() {

        /*
         * f = 6
         * g = 2
         * q = 3
         *
         * Validation on [0,10].
         */

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        ChebyshevDivisionClient.DivisionResult result =
                client.divideAndValidate(
                        new double[]{6.0},
                        new double[]{2.0},
                        0.0,
                        10.0,
                        100
                );

        assertTrue(
                result.isValid()
        );

        assertEquals(
                0.0,
                result.getResidual(),
                1e-12
        );
    }

    @Test
    void testInvalidDivisionBecauseDenominatorHasRoot() {

        /*
         * g(x) = T1(x) = x
         *
         * Root exists at x = 0.
         */

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        ChebyshevDivisionClient.DivisionResult result =
                client.divideAndValidate(
                        new double[]{1.0},
                        new double[]{0.0, 1.0},
                        101
                );

        assertFalse(
                result.isValid()
        );
    }

    @Test
    void testDivisionResultIsDefensiveCopy() {

        ChebyshevDivisionClient.DivisionResult result =
                new ChebyshevDivisionClient.DivisionResult(
                        new double[]{1.0, 2.0},
                        0.0,
                        true
                );

        double[] quotient =
                result.getQuotient();

        quotient[0] = 100.0;

        assertEquals(
                1.0,
                result.getQuotient()[0],
                1e-12
        );
    }

    @Test
    void testInvalidConfiguration() {

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChebyshevDivisionClient(
                                0,
                                1e-10
                        )
        );

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        new ChebyshevDivisionClient(
                                10,
                                0.0
                        )
        );
    }

    @Test
    void testInvalidCoefficients() {

        ChebyshevDivisionClient client =
                new ChebyshevDivisionClient();

        assertThrows(
                IllegalArgumentException.class,
                () ->
                        client.divide(
                                null,
                                new double[]{1.0}
                        )
        );
    }
}