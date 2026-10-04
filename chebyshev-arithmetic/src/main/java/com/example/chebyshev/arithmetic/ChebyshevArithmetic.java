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

    /**
     * Performs polynomial division in the Chebyshev basis.
     *
     * Returns:
     *
     * f(x) = g(x) * q(x) + r(x)
     *
     * where:
     * q(x) is the quotient
     * r(x) is the remainder
     */
    public static DivisionResult divideWithRemainder(
            double[] f,
            double[] g
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        double[] dividend =
                trimTrailingZeros(f);

        double[] divisor =
                trimTrailingZeros(g);

        if (isZeroPolynomial(divisor)) {
            throw new IllegalArgumentException(
                    "Division by zero polynomial is not allowed"
            );
        }

        if (degree(dividend) < degree(divisor)) {
            return new DivisionResult(
                    new double[]{0.0},
                    dividend.clone()
            );
        }

        double[] dividendPower =
                chebyshevToPower(dividend);

        double[] divisorPower =
                chebyshevToPower(divisor);

        double[][] division =
                dividePowerPolynomials(
                        dividendPower,
                        divisorPower
                );

        double[] quotient =
                powerToChebyshev(
                        division[0]
                );

        double[] remainder =
                powerToChebyshev(
                        division[1]
                );

        return new DivisionResult(
                quotient,
                remainder
        );
    }

    private static int degree(
            double[] coefficients
    ) {
        return trimTrailingZeros(
                coefficients
        ).length - 1;
    }

    private static boolean isZeroPolynomial(
            double[] coefficients
    ) {
        return coefficients.length == 1
                && Math.abs(coefficients[0]) < 1e-14;
    }
    private static double[] chebyshevToPower(
            double[] coefficients
    ) {
        int n = coefficients.length - 1;

        double[][] chebyshev =
                new double[n + 1][];

        /*
         * T0 = 1
         */
        chebyshev[0] =
                new double[]{1.0};

        if (n >= 1) {

            /*
             * T1 = x
             */
            chebyshev[1] =
                    new double[]{0.0, 1.0};
        }

        /*
         * Tn = 2x*T(n-1) - T(n-2)
         */
        for (int k = 2; k <= n; k++) {

            double[] current =
                    new double[k + 1];

            for (int j = 0;
                 j < chebyshev[k - 1].length;
                 j++) {

                current[j + 1] +=
                        2.0 * chebyshev[k - 1][j];
            }

            for (int j = 0;
                 j < chebyshev[k - 2].length;
                 j++) {

                current[j] -=
                        chebyshev[k - 2][j];
            }

            chebyshev[k] = current;
        }

        double[] result =
                new double[n + 1];

        for (int k = 0; k <= n; k++) {

            for (int j = 0;
                 j < chebyshev[k].length;
                 j++) {

                result[j] +=
                        coefficients[k]
                                * chebyshev[k][j];
            }
        }

        return trimTrailingZeros(result);
    }

    private static double[] powerToChebyshev(
            double[] coefficients
    ) {
        double[] remaining =
                coefficients.clone();

        int n = remaining.length - 1;

        double[] result =
                new double[n + 1];

        /*
         * Process highest powers first.
         */
        for (int k = n; k >= 0; k--) {

            if (Math.abs(remaining[k]) < 1e-14) {
                continue;
            }

            double leadingCoefficient;

            if (k == 0) {
                leadingCoefficient = 1.0;
            } else {
                /*
                 * Leading coefficient of T_k:
                 *
                 * 2^(k-1)
                 */
                leadingCoefficient =
                        Math.pow(2.0, k - 1);
            }

            double factor =
                    remaining[k]
                            / leadingCoefficient;

            result[k] += factor;

            /*
             * Subtract factor * T_k.
             */
            double[] tk =
                    chebyshevPolynomialInPowerBasis(k);

            for (int j = 0;
                 j < tk.length;
                 j++) {

                remaining[j] -=
                        factor * tk[j];
            }
        }

        return trimTrailingZeros(result);
    }

    private static double[] chebyshevPolynomialInPowerBasis(
            int n
    ) {
        if (n == 0) {
            return new double[]{1.0};
        }

        if (n == 1) {
            return new double[]{0.0, 1.0};
        }

        double[] previousPrevious =
                new double[]{1.0};

        double[] previous =
                new double[]{0.0, 1.0};

        for (int k = 2; k <= n; k++) {

            double[] current =
                    new double[k + 1];

            for (int j = 0;
                 j < previous.length;
                 j++) {

                current[j + 1] +=
                        2.0 * previous[j];
            }

            for (int j = 0;
                 j < previousPrevious.length;
                 j++) {

                current[j] -=
                        previousPrevious[j];
            }

            previousPrevious = previous;
            previous = current;
        }

        return previous;
    }

    private static double[][] dividePowerPolynomials(
            double[] dividend,
            double[] divisor
    ) {
        double[] remainder =
                dividend.clone();

        int dividendDegree =
                dividend.length - 1;

        int divisorDegree =
                divisor.length - 1;

        int quotientDegree =
                dividendDegree - divisorDegree;

        double[] quotient =
                new double[quotientDegree + 1];

        double divisorLeading =
                divisor[divisorDegree];

        if (Math.abs(divisorLeading) < 1e-14) {
            throw new IllegalArgumentException(
                    "Divisor leading coefficient must not be zero"
            );
        }

        for (int k = dividendDegree;
             k >= divisorDegree;
             k--) {

            double factor =
                    remainder[k]
                            / divisorLeading;

            int quotientIndex =
                    k - divisorDegree;

            quotient[quotientIndex] =
                    factor;

            for (int j = 0;
                 j <= divisorDegree;
                 j++) {

                remainder[
                        j + quotientIndex
                        ] -= factor * divisor[j];
            }
        }

        double[] trimmedRemainder =
                trimTrailingZeros(remainder);

        /*
         * If the remainder is numerically zero,
         * return exactly [0].
         */
        if (trimmedRemainder.length == 1
                && Math.abs(
                trimmedRemainder[0]
        ) < 1e-12) {

            trimmedRemainder =
                    new double[]{0.0};
        }

        return new double[][]{
                trimTrailingZeros(quotient),
                trimmedRemainder
        };
    }

    public static final class DivisionResult {

        private final double[] quotient;
        private final double[] remainder;

        private DivisionResult(
                double[] quotient,
                double[] remainder
        ) {
            this.quotient =
                    quotient.clone();

            this.remainder =
                    remainder.clone();
        }

        public double[] getQuotient() {
            return quotient.clone();
        }

        public double[] getRemainder() {
            return remainder.clone();
        }
    }


}