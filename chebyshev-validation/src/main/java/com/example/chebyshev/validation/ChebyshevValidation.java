package com.example.chebyshev.validation;

import com.example.chebyshev.core.ChebyshevCore;
import com.example.chebyshev.transform.ChebyshevTransform;

import java.util.Arrays;

public final class ChebyshevValidation {

    private ChebyshevValidation() {
        // Utility class
    }

    /**
     * Removes trailing coefficients whose absolute value
     * is smaller than the specified tolerance.
     *
     * Example:
     *
     * [1.0, 0.5, 0.000000000001]
     *
     * with tolerance 1e-10 becomes:
     *
     * [1.0, 0.5]
     */
    public static double[] adaptiveTruncate(
            double[] coefficients,
            double tolerance
    ) {
        validateCoefficients(coefficients);
        validateTolerance(tolerance);

        int last =
                coefficients.length - 1;

        while (last > 0
                && Math.abs(coefficients[last])
                < tolerance) {

            last--;
        }

        return Arrays.copyOf(
                coefficients,
                last + 1
        );
    }

    /**
     * Calculates the residual:
     *
     * R(x) = f(x) - g(x)q(x)
     *
     * where f, g and q are Chebyshev series.
     */
    public static double residual(
            double[] f,
            double[] g,
            double[] q,
            double x
    ) {
        validateCoefficients(f);
        validateCoefficients(g);
        validateCoefficients(q);

        validateX(x);

        double fValue =
                ChebyshevCore.clenshawT(
                        f,
                        x
                );

        double gValue =
                ChebyshevCore.clenshawT(
                        g,
                        x
                );

        double qValue =
                ChebyshevCore.clenshawT(
                        q,
                        x
                );

        return fValue
                - gValue * qValue;
    }

    /**
     * Calculates the maximum absolute residual
     * over a set of points.
     */
    public static double maxResidual(
            double[] f,
            double[] g,
            double[] q,
            double[] points
    ) {
        validateCoefficients(f);
        validateCoefficients(g);
        validateCoefficients(q);

        if (points == null
                || points.length == 0) {

            throw new IllegalArgumentException(
                    "Points must not be null or empty"
            );
        }

        double max =
                0.0;

        for (double x : points) {

            double r =
                    Math.abs(
                            residual(
                                    f,
                                    g,
                                    q,
                                    x
                            )
                    );

            max =
                    Math.max(max, r);
        }

        return max;
    }

    /**
     * Creates equally spaced validation points
     * on [a,b].
     */
    public static double[] samplePoints(
            double a,
            double b,
            int samples
    ) {
        validateInterval(a, b);

        if (samples < 2) {
            throw new IllegalArgumentException(
                    "At least two samples are required"
            );
        }

        double[] points =
                new double[samples];

        for (int i = 0;
             i < samples;
             i++) {

            double t =
                    (double) i
                            / (samples - 1);

            points[i] =
                    a + t * (b - a);
        }

        return points;
    }

    /**
     * Checks whether g(x) has a possible root
     * on [a,b].
     *
     * The interval is sampled and sign changes
     * are detected.
     */
    public static boolean hasPossibleRoot(
            double[] g,
            double a,
            double b,
            int samples
    ) {
        validateCoefficients(g);
        validateInterval(a, b);

        if (samples < 2) {
            throw new IllegalArgumentException(
                    "At least two samples are required"
            );
        }

        double previous =
                evaluateOnInterval(
                        g,
                        a,
                        a,
                        b
                );

        if (Math.abs(previous) < 1e-12) {
            return true;
        }

        for (int i = 1;
             i < samples;
             i++) {

            double x =
                    a
                            + (b - a)
                            * i
                            / (samples - 1.0);

            double current =
                    ChebyshevCore.clenshawT(
                            g,
                            ChebyshevTransform.toCanonical(
                                    x,
                                    a,
                                    b
                            )
                    );

            if (Math.abs(current) < 1e-12) {
                return true;
            }

            if (previous * current < 0.0) {
                return true;
            }

            previous = current;
        }

        return false;
    }

    /**
     * Validates a division result by checking:
     *
     * 1. denominator does not appear to have
     *    a root on [a,b]
     *
     * 2. maximum residual <= tolerance
     */
    public static boolean isValidDivision(
            double[] f,
            double[] g,
            double[] q,
            double a,
            double b,
            int samples,
            double tolerance
    ) {
        validateCoefficients(f);
        validateCoefficients(g);
        validateCoefficients(q);
        validateInterval(a, b);
        validateTolerance(tolerance);

        if (samples < 2) {
            throw new IllegalArgumentException(
                    "At least two samples are required"
            );
        }

        /*
         * Division by a function which is zero
         * somewhere in the interval is considered
         * invalid.
         */
        if (hasPossibleRoot(
                g,
                a,
                b,
                samples
        )) {
            return false;
        }

        double[] points =
                samplePoints(
                        a,
                        b,
                        samples
                );

        /*
         * Convert physical points [a,b]
         * into canonical points [-1,1].
         */
        double[] canonicalPoints =
                new double[points.length];

        for (int i = 0;
             i < points.length;
             i++) {

            canonicalPoints[i] =
                    ChebyshevTransform.toCanonical(
                            points[i],
                            a,
                            b
                    );
        }

        double error =
                maxResidual(
                        f,
                        g,
                        q,
                        canonicalPoints
                );

        return error <= tolerance;
    }

    /**
     * Returns the maximum residual of a division
     * on [a,b].
     */
    public static double divisionError(
            double[] f,
            double[] g,
            double[] q,
            double a,
            double b,
            int samples
    ) {
        validateInterval(a, b);

        double[] points =
                samplePoints(
                        a,
                        b,
                        samples
                );

        double[] canonicalPoints =
                new double[points.length];

        for (int i = 0;
             i < points.length;
             i++) {

            canonicalPoints[i] =
                    ChebyshevTransform.toCanonical(
                            points[i],
                            a,
                            b
                    );
        }

        return maxResidual(
                f,
                g,
                q,
                canonicalPoints
        );
    }

    private static double evaluateOnInterval(
            double[] coefficients,
            double x,
            double a,
            double b
    ) {
        double canonical =
                ChebyshevTransform.toCanonical(
                        x,
                        a,
                        b
                );

        return ChebyshevCore.clenshawT(
                coefficients,
                canonical
        );
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

    private static void validateTolerance(
            double tolerance
    ) {
        if (!Double.isFinite(tolerance)
                || tolerance <= 0.0) {

            throw new IllegalArgumentException(
                    "Tolerance must be positive"
            );
        }
    }

    private static void validateX(
            double x
    ) {
        if (!Double.isFinite(x)
                || x < -1.0
                || x > 1.0) {

            throw new IllegalArgumentException(
                    "x must be in [-1,1]"
            );
        }
    }
}