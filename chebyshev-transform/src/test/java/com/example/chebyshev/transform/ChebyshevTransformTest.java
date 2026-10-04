package com.example.chebyshev.transform;

import org.junit.jupiter.api.Test;

import static org.junit.jupiter.api.Assertions.*;

class ChebyshevTransformTest {

    @Test
    void testToCanonical() {
        assertEquals(
                -1.0,
                ChebyshevTransform.toCanonical(
                        0.0,
                        0.0,
                        10.0
                ),
                1e-12
        );

        assertEquals(
                1.0,
                ChebyshevTransform.toCanonical(
                        10.0,
                        0.0,
                        10.0
                ),
                1e-12
        );

        assertEquals(
                0.0,
                ChebyshevTransform.toCanonical(
                        5.0,
                        0.0,
                        10.0
                ),
                1e-12
        );
    }

    @Test
    void testFromCanonical() {
        assertEquals(
                0.0,
                ChebyshevTransform.fromCanonical(
                        -1.0,
                        0.0,
                        10.0
                ),
                1e-12
        );

        assertEquals(
                10.0,
                ChebyshevTransform.fromCanonical(
                        1.0,
                        0.0,
                        10.0
                ),
                1e-12
        );

        assertEquals(
                5.0,
                ChebyshevTransform.fromCanonical(
                        0.0,
                        0.0,
                        10.0
                ),
                1e-12
        );
    }

    @Test
    void testMappingIsInverse() {
        double a = 2.0;
        double b = 8.0;

        for (double x = 2.0;
             x <= 8.0;
             x += 0.5) {

            double t =
                    ChebyshevTransform.toCanonical(
                            x,
                            a,
                            b
                    );

            double restored =
                    ChebyshevTransform.fromCanonical(
                            t,
                            a,
                            b
                    );

            assertEquals(
                    x,
                    restored,
                    1e-12
            );
        }
    }

    @Test
    void testFirstKindNodes() {
        double[] nodes =
                ChebyshevTransform.firstKindNodes(4);

        assertEquals(4, nodes.length);

        for (double x : nodes) {
            assertTrue(x >= -1.0);
            assertTrue(x <= 1.0);
        }

        assertEquals(
                Math.cos(Math.PI / 8.0),
                nodes[0],
                1e-12
        );
    }

    @Test
    void testSecondKindNodes() {
        double[] nodes =
                ChebyshevTransform.secondKindNodes(4);

        assertEquals(5, nodes.length);

        assertEquals(
                1.0,
                nodes[0],
                1e-12
        );

        assertEquals(
                -1.0,
                nodes[4],
                1e-12
        );
    }

    @Test
    void testMapNodes() {
        double[] nodes = {
                -1.0,
                0.0,
                1.0
        };

        double[] mapped =
                ChebyshevTransform.mapNodes(
                        nodes,
                        10.0,
                        20.0
                );

        assertEquals(
                10.0,
                mapped[0],
                1e-12
        );

        assertEquals(
                15.0,
                mapped[1],
                1e-12
        );

        assertEquals(
                20.0,
                mapped[2],
                1e-12
        );
    }

    @Test
    void testTransformConstantFunction() {
        /*
         * f(x) = 1
         *
         * Expected:
         *
         * c0 = 1
         * all other coefficients = 0
         */
        double[] values = {
                1.0,
                1.0,
                1.0,
                1.0,
                1.0
        };

        double[] coefficients =
                ChebyshevTransform.transform(values);

        assertEquals(
                1.0,
                coefficients[0],
                1e-12
        );

        for (int i = 1;
             i < coefficients.length;
             i++) {

            assertEquals(
                    0.0,
                    coefficients[i],
                    1e-12
            );
        }
    }

    @Test
    void testEvaluate() {
        /*
         * f(x) = 1 + 2T1(x) + 3T2(x)
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
                ChebyshevTransform.evaluate(
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
    void testInvalidInterval() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevTransform.toCanonical(
                        1.0,
                        5.0,
                        5.0
                )
        );
    }

    @Test
    void testInvalidCanonicalValue() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevTransform.fromCanonical(
                        2.0,
                        0.0,
                        10.0
                )
        );
    }

    @Test
    void testInvalidNodes() {
        assertThrows(
                IllegalArgumentException.class,
                () -> ChebyshevTransform.mapNodes(
                        new double[]{-2.0},
                        0.0,
                        1.0
                )
        );
    }

    @Test
    void testTransformLinearFunction() {
        /*
         * f(x) = x = T1(x)
         *
         * Values on Lobatto nodes for n = 4:
         *
         * x = 1, sqrt(2)/2, 0, -sqrt(2)/2, -1
         */
        double[] values = {
                1.0,
                Math.sqrt(2.0) / 2.0,
                0.0,
                -Math.sqrt(2.0) / 2.0,
                -1.0
        };

        double[] coefficients =
                ChebyshevTransform.transform(values);

        assertEquals(
                0.0,
                coefficients[0],
                1e-12
        );

        assertEquals(
                1.0,
                coefficients[1],
                1e-12
        );

        assertEquals(
                0.0,
                coefficients[2],
                1e-12
        );

        assertEquals(
                0.0,
                coefficients[3],
                1e-12
        );

        assertEquals(
                0.0,
                coefficients[4],
                1e-12
        );
    }

    @Test
    void testTransformQuadraticFunction() {
        /*
         * x² = (T2(x) + T0(x)) / 2
         */
        double[] values = new double[5];

        double[] nodes =
                ChebyshevTransform.secondKindNodes(4);

        for (int i = 0; i < nodes.length; i++) {
            values[i] =
                    nodes[i] * nodes[i];
        }

        double[] coefficients =
                ChebyshevTransform.transform(values);

        assertEquals(
                0.5,
                coefficients[0],
                1e-12
        );

        assertEquals(
                0.0,
                coefficients[1],
                1e-12
        );

        assertEquals(
                0.5,
                coefficients[2],
                1e-12
        );
    }

    @Test
    void testTransformAndEvaluate() {
        /*
         * f(x) = 1 + 2x + 3x²
         */
        double[] values = new double[9];

        double[] nodes =
                ChebyshevTransform.secondKindNodes(8);

        for (int i = 0; i < nodes.length; i++) {

            double x = nodes[i];

            values[i] =
                    1.0
                            + 2.0 * x
                            + 3.0 * x * x;
        }

        double[] coefficients =
                ChebyshevTransform.transform(values);

        double x = 0.37;

        double expected =
                1.0
                        + 2.0 * x
                        + 3.0 * x * x;

        double actual =
                ChebyshevTransform.evaluate(
                        coefficients,
                        x
                );

        assertEquals(
                expected,
                actual,
                1e-10
        );
    }

    @Test
    void testInverseTransformConstant() {

        double[] coefficients = {
                5.0,
                0.0,
                0.0,
                0.0,
                0.0
        };

        double[] values =
                ChebyshevTransform.inverseTransform(coefficients);

        for (double value : values) {
            assertEquals(5.0, value, 1e-12);
        }
    }

    @Test
    void testInverseTransformLinear() {

        double[] coefficients = {
                0.0,
                1.0,
                0.0,
                0.0,
                0.0
        };

        double[] values =
                ChebyshevTransform.inverseTransform(coefficients);

        double[] nodes =
                ChebyshevTransform.secondKindNodes(4);

        for (int i = 0; i < nodes.length; i++) {
            assertEquals(
                    nodes[i],
                    values[i],
                    1e-12
            );
        }
    }

    @Test
    void testTransformInverseRoundTrip() {

        double[] originalCoefficients = {
                1.0,
                2.0,
                0.5,
                -1.0,
                0.25
        };

        double[] values =
                ChebyshevTransform.inverseTransform(
                        originalCoefficients
                );

        double[] recoveredCoefficients =
                ChebyshevTransform.transform(values);

        for (int i = 0;
             i < originalCoefficients.length;
             i++) {

            assertEquals(
                    originalCoefficients[i],
                    recoveredCoefficients[i],
                    1e-10
            );
        }
    }
}