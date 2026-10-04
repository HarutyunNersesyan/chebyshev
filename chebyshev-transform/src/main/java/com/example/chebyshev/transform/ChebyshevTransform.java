package com.example.chebyshev.transform;

import com.example.chebyshev.core.ChebyshevCore;
import org.apache.commons.math3.complex.Complex;
import org.apache.commons.math3.transform.DftNormalization;
import org.apache.commons.math3.transform.FastFourierTransformer;
import org.apache.commons.math3.transform.TransformType;

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

        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException(
                    "x must be finite"
            );
        }

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

        if (!Double.isFinite(t)) {
            throw new IllegalArgumentException(
                    "t must be finite"
            );
        }

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

            if (!Double.isFinite(nodes[i])) {
                throw new IllegalArgumentException(
                        "Node must be finite"
                );
            }

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
     * Computes Chebyshev coefficients from
     * function values on Chebyshev-Lobatto nodes.
     *
     * The nodes are:
     *
     * x_j = cos(jπ/n), j = 0,...,n
     *
     * The transform is implemented using a DCT-I
     * represented through an FFT of length 2n.
     *
     * Complexity: O(N log N).
     *
     * The returned coefficients satisfy:
     *
     * f(x) = Σ c_k T_k(x)
     */
    public static double[] transform(
            double[] values
    ) {
        validateValues(values);

        int n = values.length - 1;

        /*
         * DCT-I can be represented as an FFT
         * of an even extension.
         *
         * FFT length = 2n.
         */
        double[] extended =
                new double[2 * n];

        /*
         * First half:
         *
         * y_j = f_j
         *
         * j = 0,...,n
         */
        for (int j = 0; j <= n; j++) {
            extended[j] = values[j];
        }

        /*
         * Second half:
         *
         * y_{2n-j} = f_j
         *
         * j = 1,...,n-1
         */
        for (int j = 1; j < n; j++) {
            extended[2 * n - j] =
                    values[j];
        }

        FastFourierTransformer fft =
                new FastFourierTransformer(
                        DftNormalization.STANDARD
                );

        Complex[] spectrum =
                fft.transform(
                        extended,
                        TransformType.FORWARD
                );

        double[] coefficients =
                new double[n + 1];

        /*
         * For the even extension:
         *
         * Re(FFT[k]) =
         *
         * f0 + (-1)^k fn
         * + 2 Σ f_j cos(πjk/n)
         *
         * Therefore the DCT-I coefficient is:
         *
         * c_k = Re(FFT[k]) / n
         *
         * with endpoint coefficients divided by 2.
         */
        for (int k = 0; k <= n; k++) {

            coefficients[k] =
                    spectrum[k].getReal() / n;
        }

        /*
         * Chebyshev series convention:
         *
         * f(x) =
         * c0*T0(x)
         * + c1*T1(x)
         * + ...
         *
         * DCT-I gives endpoint coefficients
         * with the usual factor of 2.
         */
        coefficients[0] /= 2.0;
        coefficients[n] /= 2.0;

        return coefficients;
    }

    private static void validateFftSize(int n) {
        if (n <= 0 || (n & (n - 1)) != 0) {
            throw new IllegalArgumentException(
                    "The number of intervals must be a power of two"
            );
        }
    }

    /**
     * Computes function values on Chebyshev-Lobatto
     * nodes from Chebyshev coefficients.
     *
     * The input coefficients are:
     *
     * f(x) = Σ c_k T_k(x)
     *
     * The returned values correspond to:
     *
     * x_j = cos(jπ/n)
     *
     * j = 0,...,n
     *
     * Complexity: O(N log N).
     */
    public static double[] inverseTransform(double[] coefficients) {
        validateCoefficients(coefficients);

        int n = coefficients.length - 1;

        if (n == 0) {
            return new double[]{coefficients[0]};
        }

        validateFftSize(n);

        Complex[] spectrum = new Complex[2 * n];

        spectrum[0] =
                new Complex(2.0 * n * coefficients[0], 0.0);

        spectrum[n] =
                new Complex(2.0 * n * coefficients[n], 0.0);

        for (int k = 1; k < n; k++) {

            Complex value =
                    new Complex(n * coefficients[k], 0.0);

            spectrum[k] = value;
            spectrum[2 * n - k] = value;
        }

        FastFourierTransformer fft =
                new FastFourierTransformer(
                        DftNormalization.STANDARD
                );

        Complex[] result =
                fft.transform(
                        spectrum,
                        TransformType.INVERSE
                );

        double[] values = new double[n + 1];

        for (int j = 0; j <= n; j++) {
            values[j] = result[j].getReal();
        }

        return values;
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
        validateCoefficients(coefficients);

        if (!Double.isFinite(x)) {
            throw new IllegalArgumentException(
                    "x must be finite"
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

    private static void validateValues(
            double[] values
    ) {
        if (values == null
                || values.length < 2) {

            throw new IllegalArgumentException(
                    "At least two values are required"
            );
        }

        for (double value : values) {
            if (!Double.isFinite(value)) {
                throw new IllegalArgumentException(
                        "Values must be finite"
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