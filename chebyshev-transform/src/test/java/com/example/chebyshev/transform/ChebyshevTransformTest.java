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
}