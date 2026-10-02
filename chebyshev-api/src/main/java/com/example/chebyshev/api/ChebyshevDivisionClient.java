package com.example.chebyshev.api;

import com.example.chebyshev.arithmetic.ChebyshevArithmetic;
import com.example.chebyshev.core.ChebyshevCore;
import com.example.chebyshev.solver.ChebyshevMatrixSolver;
import com.example.chebyshev.transform.ChebyshevTransform;
import com.example.chebyshev.validation.ChebyshevValidation;

/**
 * Public facade API for Chebyshev polynomial division.
 *
 * This class combines:
 * - Chebyshev arithmetic
 * - direct matrix-based division
 * - validation
 * - residual calculation
 * - denominator root checking
 */
public class ChebyshevDivisionClient {

    private final int iterations;
    private final double tolerance;

    /**
     * Creates a client with default configuration.
     *
     * Default:
     * iterations = 10
     * tolerance = 1e-10
     */
    public ChebyshevDivisionClient() {
        this(10, 1e-10);
    }

    /**
     * Creates a client with custom configuration.
     *
     * @param iterations number of Newton-Schulz iterations
     * @param tolerance validation tolerance
     */
    public ChebyshevDivisionClient(
            int iterations,
            double tolerance
    ) {
        if (iterations <= 0) {
            throw new IllegalArgumentException(
                    "iterations must be positive"
            );
        }

        if (!Double.isFinite(tolerance) || tolerance <= 0.0) {
            throw new IllegalArgumentException(
                    "tolerance must be positive and finite"
            );
        }

        this.iterations = iterations;
        this.tolerance = tolerance;
    }

    /**
     * Divides f(x) by g(x) using Newton-Schulz inversion.
     *
     * @param f numerator coefficients
     * @param g denominator coefficients
     * @return quotient coefficients
     */
    public double[] divide(
            double[] f,
            double[] g
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        return ChebyshevArithmetic.divide(
                f,
                g,
                iterations
        );
    }

    /**
     * Divides and validates the result on [-1, 1].
     *
     * Before division, the denominator is checked for
     * possible roots on [-1, 1].
     *
     * @param f numerator coefficients
     * @param g denominator coefficients
     * @param samples number of validation points
     * @return division result
     */
    public DivisionResult divideAndValidate(
            double[] f,
            double[] g,
            int samples
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        /*
         * IMPORTANT:
         * Check the denominator before attempting division.
         *
         * Example:
         * g = {0, 1}
         *
         * means:
         * g(x) = T1(x) = x
         *
         * which has a root at x = 0.
         *
         * If we call divide() first, Newton-Schulz inversion
         * may reject g[0] == 0 before validation can report
         * the invalid denominator.
         */
        boolean denominatorValid =
                !ChebyshevValidation.hasPossibleRoot(
                        g,
                        -1.0,
                        1.0,
                        samples
                );

        if (!denominatorValid) {
            return new DivisionResult(
                    new double[]{0.0},
                    Double.POSITIVE_INFINITY,
                    false
            );
        }

        double[] quotient =
                divide(f, g);

        double[] points =
                ChebyshevValidation.samplePoints(
                        -1.0,
                        1.0,
                        samples
                );

        double residual =
                ChebyshevValidation.maxResidual(
                        f,
                        g,
                        quotient,
                        points
                );

        boolean valid =
                residual <= tolerance;

        return new DivisionResult(
                quotient,
                residual,
                valid
        );
    }

    /**
     * Divides and validates the result on a physical interval [a, b].
     *
     * The physical interval is mapped to the canonical
     * Chebyshev interval [-1, 1] for evaluation.
     *
     * @param f numerator coefficients
     * @param g denominator coefficients
     * @param a left endpoint
     * @param b right endpoint
     * @param samples number of validation points
     * @return division result
     */
    public DivisionResult divideAndValidate(
            double[] f,
            double[] g,
            double a,
            double b,
            int samples
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        if (!Double.isFinite(a) || !Double.isFinite(b)) {
            throw new IllegalArgumentException(
                    "Interval endpoints must be finite"
            );
        }

        if (a >= b) {
            throw new IllegalArgumentException(
                    "Require a < b"
            );
        }

        /*
         * Check denominator roots BEFORE division.
         */
        boolean denominatorValid =
                !ChebyshevValidation.hasPossibleRoot(
                        g,
                        a,
                        b,
                        samples
                );

        if (!denominatorValid) {
            return new DivisionResult(
                    new double[]{0.0},
                    Double.POSITIVE_INFINITY,
                    false
            );
        }

        double[] quotient =
                divide(f, g);

        /*
         * Generate points in the physical interval [a, b].
         */
        double[] physicalPoints =
                ChebyshevValidation.samplePoints(
                        a,
                        b,
                        samples
                );

        /*
         * Convert physical points to canonical Chebyshev
         * coordinates [-1, 1].
         */
        double[] canonicalPoints =
                new double[physicalPoints.length];

        for (int i = 0;
             i < physicalPoints.length;
             i++) {

            canonicalPoints[i] =
                    ChebyshevTransform.toCanonical(
                            physicalPoints[i],
                            a,
                            b
                    );
        }

        /*
         * Calculate the residual in canonical coordinates.
         */
        double residual =
                ChebyshevValidation.maxResidual(
                        f,
                        g,
                        quotient,
                        canonicalPoints
                );

        boolean valid =
                residual <= tolerance;

        return new DivisionResult(
                quotient,
                residual,
                valid
        );
    }

    /**
     * Performs direct polynomial division using
     * the Chebyshev matrix representation.
     *
     * @param f numerator coefficients
     * @param g denominator coefficients
     * @param degree resulting quotient degree
     * @return quotient coefficients
     */
    public double[] divideDirect(
            double[] f,
            double[] g,
            int degree
    ) {
        validateCoefficients(f);
        validateCoefficients(g);

        if (degree < 0) {
            throw new IllegalArgumentException(
                    "degree must be non-negative"
            );
        }

        return ChebyshevMatrixSolver.divide(
                f,
                g,
                degree
        );
    }

    /**
     * Returns the configured number of iterations.
     */
    public int getIterations() {
        return iterations;
    }

    /**
     * Returns the configured validation tolerance.
     */
    public double getTolerance() {
        return tolerance;
    }

    /**
     * Validates a coefficient array.
     */
    private void validateCoefficients(
            double[] coefficients
    ) {
        if (coefficients == null) {
            throw new IllegalArgumentException(
                    "Coefficients must not be null"
            );
        }

        if (coefficients.length == 0) {
            throw new IllegalArgumentException(
                    "Coefficients must not be empty"
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
     * Immutable result of a division operation.
     */
    public static final class DivisionResult {

        private final double[] quotient;
        private final double residual;
        private final boolean valid;

        public DivisionResult(
                double[] quotient,
                double residual,
                boolean valid
        ) {
            if (quotient == null) {
                throw new IllegalArgumentException(
                        "quotient must not be null"
                );
            }

            /*
             * Defensive copy prevents external modification
             * of the internal quotient array.
             */
            this.quotient =
                    quotient.clone();

            this.residual = residual;
            this.valid = valid;
        }

        /**
         * Returns a defensive copy of the quotient.
         */
        public double[] getQuotient() {
            return quotient.clone();
        }

        /**
         * Returns the maximum residual.
         */
        public double getResidual() {
            return residual;
        }

        /**
         * Returns whether the division passed validation.
         */
        public boolean isValid() {
            return valid;
        }

        @Override
        public String toString() {
            return "DivisionResult{" +
                    "quotient=" +
                    java.util.Arrays.toString(quotient) +
                    ", residual=" +
                    residual +
                    ", valid=" +
                    valid +
                    '}';
        }
    }
}