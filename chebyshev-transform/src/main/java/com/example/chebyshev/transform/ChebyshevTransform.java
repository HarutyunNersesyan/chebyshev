package com.example.chebyshev.transform;

import com.example.chebyshev.core.ChebyshevCore;

public final class ChebyshevTransform {

    private ChebyshevTransform() {
        // Utility class
    }

    /**
     * Maps x from [a, b] to t in [-1, 1].
     *
     * t = (2x - (a + b)) / (b - a)
     */
    public static double toCanonical(
            double x,
            double a,
            double b
    ) {
        validateInterval(a, b);

        return (2.0 * x - (a + b))
                / (b - a);
    }

    /**
     * Maps t from [-1, 1] to x in [a, b].
     *
     * x = ((b-a)t + (a+b)) / 2
     */
    public static double fromCanonical(
            double t,
            double a,
            double b
    ) {
        validateInterval(a, b);

        if (t < -1.0 || t > 1.0) {
            throw new IllegalArgumentException(
                    "t must be in [-1, 1]"
            );
        }

        return ((b - a) * t + (a + b))
                / 2.0;
    }

    /**
     * Returns roots of T_n(x).
     *
     * x_k = cos((2k + 1)π / (2n))
     *
     * k = 0,...,n-1
     */
    public static double[] firstKindNodes(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException(
                    "n must be positive"
            );
        }

        double[] nodes = new double[n];

        for (int k = 0; k < n; k++) {
            nodes[k] = Math.cos(
                    (2.0 * k + 1.0)
                            * Math.PI
                            / (2.0 * n)
            );
        }

        return nodes;
    }

    /**
     * Returns Chebyshev-Lobatto nodes.
     *
     * x_k = cos(kπ/n)
     *
     * k = 0,...,n
     *
     * These are extrema nodes of T_n.
     */
    public static double[] secondKindNodes(int n) {
        if (n <= 0) {
            throw new IllegalArgumentException(
                    "n must be positive"
            );
        }

        double[] nodes = new double[n + 1];

        for (int k = 0; k <= n; k++) {
            nodes[k] = Math.cos(
                    k * Math.PI / n
            );
        }

        return nodes;
    }

    /**
     * Maps canonical nodes [-1,1]
     * into physical interval [a,b].
     */
    public static double[] mapNodes(
            double[] nodes,
            double a,
            double b
    ) {
        validateInterval(a, b);

        if (nodes == null || nodes.length == 0) {
            throw new IllegalArgumentException(
                    "Nodes must not be null or empty"
            );
        }

        double[] result =
                new double[nodes.length];

        for (int i = 0; i < nodes.length; i++) {

            if (nodes[i] < -1.0
                    || nodes[i] > 1.0) {

                throw new IllegalArgumentException(
                        "Node must be in [-1,1]"
                );
            }

            result[i] =
                    fromCanonical(
                            nodes[i],
                            a,
                            b
                    );
        }

        return result;
    }

    /**
     * Computes Chebyshev coefficients
     * from function values on Lobatto nodes.
     *
     * This implementation uses the DCT-I formula.
     *
     * Complexity: O(N²).
     *
     * It is intentionally kept as a mathematically
     * stable reference implementation.
     */
    public static double[] transform(
            double[] values
    ) {
        if (values == null
                || values.length < 2) {

            throw new IllegalArgumentException(
                    "At least two values are required"
            );
        }

        int n = values.length - 1;

        double[] coefficients =
                new double[n + 1];

        for (int k = 0; k <= n; k++) {

            double sum = 0.0;

            for (int j = 0; j <= n; j++) {

                double angle =
                        Math.PI * j * k / n;

                double weight = 1.0;

                if (j == 0 || j == n) {
                    weight = 0.5;
                }

                sum += weight
                        * values[j]
                        * Math.cos(angle);
            }

            coefficients[k] =
                    2.0 * sum / n;
        }

        coefficients[0] /= 2.0;
        coefficients[n] /= 2.0;

        return coefficients;
    }

    /**
     * Evaluates a Chebyshev series:
     *
     * f(x) = Σ c_k T_k(x)
     */
    public static double evaluate(
            double[] coefficients,
            double x
    ) {
        if (coefficients == null
                || coefficients.length == 0) {

            throw new IllegalArgumentException(
                    "Coefficients must not be null or empty"
            );
        }

        if (x < -1.0 || x > 1.0) {
            throw new IllegalArgumentException(
                    "x must be in [-1,1]"
            );
        }

        return ChebyshevCore.clenshawT(
                coefficients,
                x
        );
    }

    private static void validateInterval(
            double a,
            double b
    ) {
        if (!Double.isFinite(a)
                || !Double.isFinite(b)) {

            throw new IllegalArgumentException(
                    "Interval boundaries must be finite"
            );
        }

        if (a >= b) {
            throw new IllegalArgumentException(
                    "Require a < b"
            );
        }
    }
}