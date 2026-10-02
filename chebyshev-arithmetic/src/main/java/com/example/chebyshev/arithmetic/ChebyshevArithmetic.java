package com.example.chebyshev.arithmetic;

import com.example.chebyshev.core.ChebyshevCore;

import java.util.Arrays;

public final class ChebyshevArithmetic {

    private ChebyshevArithmetic() {
        // Utility class
    }

    /**
     * Adds two Chebyshev coefficient arrays.
     */
    public static double[] add(
            double[] a,
            double[] b
    ) {
        validateCoefficients(a);
        validateCoefficients(b);

        int size = Math.max(a.length, b.length);
        double[] result = new double[size];

        for (int i = 0; i < size; i++) {
            double av = i < a.length ? a[i] : 0.0;
            double bv = i < b.length ? b[i] : 0.0;

            result[i] = av + bv;
        }

        return trimTrailingZeros(result);
    }

    /**
     * Subtracts b from a.
     */
    public static double[] subtract(
            double[] a,
            double[] b
    ) {
        validateCoefficients(a);
        validateCoefficients(b);

        int size = Math.max(a.length, b.length);
        double[] result = new double[size];

        for (int i = 0; i < size; i++) {
            double av = i < a.length ? a[i] : 0.0;
            double bv = i < b.length ? b[i] : 0.0;

            result[i] = av - bv;
        }

        return trimTrailingZeros(result);
    }

    /**
     * Multiplies two Chebyshev series.
     *
     * Uses:
     *
     * 2 T_m(x) T_n(x)
     * =
     * T_{m+n}(x) + T_{|m-n|}(x)
     *
     * Therefore:
     *
     * T_m T_n
     * =
     * 1/2 [T_{m+n} + T_{|m-n|}]
     */
    public static double[] multiply(
            double[] a,
            double[] b
    ) {
        validateCoefficients(a);
        validateCoefficients(b);

        double[] result =
                new double[a.length + b.length - 1];

        for (int m = 0; m < a.length; m++) {
            for (int n = 0; n < b.length; n++) {

                double value =
                        0.5 * a[m] * b[n];

                result[m + n] += value;
                result[Math.abs(m - n)] += value;
            }
        }

        return trimTrailingZeros(result);
    }

    /**
     * Evaluates a Chebyshev series.
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

    /**
     * Computes an approximate inverse of a Chebyshev series:
     *
     * p_{k+1} = p_k (2 - g p_k)
     *
     * This is the Newton-Schulz iteration.
     *
     * The result is truncated to the degree of g.
     */
    public static double[] inverse(
            double[] g,
            int iterations,
            double initialGuess
    ) {
        validateCoefficients(g);

        if (iterations <= 0) {
            throw new IllegalArgumentException(
                    "Iterations must be positive"
            );
        }

        if (!Double.isFinite(initialGuess)) {
            throw new IllegalArgumentException(
                    "Initial guess must be finite"
            );
        }

        int size = g.length;

        double[] p = new double[size];
        p[0] = initialGuess;

        for (int iteration = 0;
             iteration < iterations;
             iteration++) {

            double[] gp =
                    multiplyTruncated(
                            g,
                            p,
                            size
                    );

            double[] twoMinusGp =
                    new double[size];

            twoMinusGp[0] = 2.0;

            for (int i = 0; i < size; i++) {
                twoMinusGp[i] -= gp[i];
            }

            p = multiplyTruncated(
                    p,
                    twoMinusGp,
                    size
            );
        }

        return trimTrailingZeros(p);
    }

    /**
     * Computes an approximate inverse using
     * 1 / g_0 as the initial approximation.
     */
    public static double[] inverse(
            double[] g,
            int iterations
    ) {
        validateCoefficients(g);

        if (g[0] == 0.0) {
            throw new IllegalArgumentException(
                    "g[0] must not be zero"
            );
        }

        return inverse(
                g,
                iterations,
                1.0 / g[0]
        );
    }

    /**
     * Divides f by g using:
     *
     * f / g = f * (1 / g)
     */
    public static double[] divide(
            double[] f,
            double[] g,
            int iterations,
            double initialGuess
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        double[] inverseG =
                inverse(
                        g,
                        iterations,
                        initialGuess
                );

        return multiply(
                f,
                inverseG
        );
    }

    /**
     * Divides f by g using 1/g[0]
     * as the initial approximation.
     */
    public static double[] divide(
            double[] f,
            double[] g,
            int iterations
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        double[] inverseG =
                inverse(
                        g,
                        iterations
                );

        return multiply(
                f,
                inverseG
        );
    }

    /**
     * Converts a T-series to a U-series.
     *
     * T_0 = U_0
     *
     * T_1 = 1/2 U_1
     *
     * T_n = 1/2 (U_n - U_{n-2}), n >= 2
     */
    public static double[] toU(
            double[] tCoefficients
    ) {
        validateCoefficients(tCoefficients);

        double[] u =
                new double[tCoefficients.length];

        for (int n = 0;
             n < tCoefficients.length;
             n++) {

            double c = tCoefficients[n];

            if (n == 0) {
                u[0] += c;
            } else if (n == 1) {
                u[1] += c / 2.0;
            } else {
                u[n] += c / 2.0;
                u[n - 2] -= c / 2.0;
            }
        }

        return trimTrailingZeros(u);
    }

    /**
     * Converts a U-series to a T-series.
     *
     * U_n =
     *
     * 2(T_n + T_{n-2} + ...)
     *
     * with a correction for the T_0 term when n is even.
     */
    public static double[] toT(
            double[] uCoefficients
    ) {
        validateCoefficients(uCoefficients);

        double[] t =
                new double[uCoefficients.length];

        for (int n = 0;
             n < uCoefficients.length;
             n++) {

            double c = uCoefficients[n];

            for (int k = n;
                 k >= 0;
                 k -= 2) {

                t[k] += 2.0 * c;
            }

            /*
             * For even n:
             *
             * U_n =
             * 2(T_n + T_{n-2} + ...)
             * - T_0
             */
            if (n % 2 == 0) {
                t[0] -= c;
            }
        }

        return trimTrailingZeros(t);
    }

    /**
     * Multiplies two series and keeps only
     * the first 'size' coefficients.
     */
    private static double[] multiplyTruncated(
            double[] a,
            double[] b,
            int size
    ) {
        double[] result =
                new double[size];

        for (int m = 0; m < a.length; m++) {
            for (int n = 0; n < b.length; n++) {

                int sumIndex = m + n;
                int diffIndex = Math.abs(m - n);

                double value =
                        0.5 * a[m] * b[n];

                if (sumIndex < size) {
                    result[sumIndex] += value;
                }

                if (diffIndex < size) {
                    result[diffIndex] += value;
                }
            }
        }

        return result;
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

    /**
     * Removes insignificant trailing zeros,
     * but always keeps at least one coefficient.
     */
    private static double[] trimTrailingZeros(
            double[] coefficients
    ) {
        int last =
                coefficients.length - 1;

        while (last > 0
                && Math.abs(coefficients[last]) < 1e-14) {
            last--;
        }

        return Arrays.copyOf(
                coefficients,
                last + 1
        );
    }
}