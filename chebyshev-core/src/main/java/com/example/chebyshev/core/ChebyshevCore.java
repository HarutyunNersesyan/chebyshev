package com.example.chebyshev.core;

public final class ChebyshevCore {

    private ChebyshevCore() {
        // Utility class
    }

    /**
     * Chebyshev polynomial of the first kind:
     *
     * T_0(x) = 1
     * T_1(x) = x
     * T_n(x) = 2xT_{n-1}(x) - T_{n-2}(x)
     */
    public static double T(int n, double x) {
        validateDegree(n);
        validateX(x);

        if (n == 0) {
            return 1.0;
        }

        if (n == 1) {
            return x;
        }

        double t0 = 1.0;
        double t1 = x;

        for (int k = 2; k <= n; k++) {
            double t2 = 2.0 * x * t1 - t0;

            t0 = t1;
            t1 = t2;
        }

        return t1;
    }

    /**
     * Chebyshev polynomial of the second kind:
     *
     * U_0(x) = 1
     * U_1(x) = 2x
     * U_n(x) = 2xU_{n-1}(x) - U_{n-2}(x)
     */
    public static double U(int n, double x) {
        validateDegree(n);
        validateX(x);

        if (n == 0) {
            return 1.0;
        }

        if (n == 1) {
            return 2.0 * x;
        }

        double u0 = 1.0;
        double u1 = 2.0 * x;

        for (int k = 2; k <= n; k++) {
            double u2 = 2.0 * x * u1 - u0;

            u0 = u1;
            u1 = u2;
        }

        return u1;
    }

    /**
     * Trigonometric representation of T_n:
     *
     * T_n(x) = cos(n * arccos(x))
     */
    public static double tTrig(int n, double x) {
        validateDegree(n);
        validateX(x);

        return Math.cos(n * Math.acos(x));
    }

    /**
     * Trigonometric representation of U_n:
     *
     * U_n(x) = sin((n+1)theta) / sin(theta)
     *
     * where x = cos(theta).
     */
    public static double uTrig(int n, double x) {
        validateDegree(n);
        validateX(x);

        /*
         * Special case x = 1:
         *
         * U_n(1) = n + 1
         */
        if (x == 1.0) {
            return n + 1.0;
        }

        /*
         * Special case x = -1:
         *
         * U_n(-1) = (-1)^n (n + 1)
         */
        if (x == -1.0) {
            return (n % 2 == 0)
                    ? n + 1.0
                    : -(n + 1.0);
        }

        double theta = Math.acos(x);

        return Math.sin((n + 1.0) * theta)
                / Math.sin(theta);
    }

    /**
     * Clenshaw evaluation for a Chebyshev T-series:
     *
     * f(x) = c_0 T_0(x)
     *      + c_1 T_1(x)
     *      + ...
     *      + c_n T_n(x)
     */
    public static double clenshawT(
            double[] coefficients,
            double x
    ) {
        validateCoefficients(coefficients);
        validateX(x);

        int n = coefficients.length - 1;

        if (n == 0) {
            return coefficients[0];
        }

        double bNext = 0.0;
        double bNextNext = 0.0;

        for (int k = n; k >= 1; k--) {

            double bCurrent =
                    2.0 * x * bNext
                            - bNextNext
                            + coefficients[k];

            bNextNext = bNext;
            bNext = bCurrent;
        }

        return coefficients[0]
                + x * bNext
                - bNextNext;
    }

    /**
     * Clenshaw evaluation for a Chebyshev U-series:
     *
     * f(x) = c_0 U_0(x)
     *      + c_1 U_1(x)
     *      + ...
     *      + c_n U_n(x)
     */
    public static double clenshawU(
            double[] coefficients,
            double x
    ) {
        validateCoefficients(coefficients);
        validateX(x);

        int n = coefficients.length - 1;

        if (n == 0) {
            return coefficients[0];
        }

        double bNext = 0.0;
        double bNextNext = 0.0;

        for (int k = n; k >= 0; k--) {

            double bCurrent =
                    2.0 * x * bNext
                            - bNextNext
                            + coefficients[k];

            bNextNext = bNext;
            bNext = bCurrent;
        }

        return bNext;
    }

    /**
     * Derivative of T_n:
     *
     * dT_n/dx = n * U_{n-1}(x)
     */
    public static double derivativeT(int n, double x) {
        validateDegree(n);
        validateX(x);

        if (n == 0) {
            return 0.0;
        }

        return n * U(n - 1, x);
    }

    /**
     * Indefinite integral of T_n(x):
     *
     * ∫T_0(x)dx = x
     *
     * ∫T_1(x)dx = x² / 2
     *
     * For n >= 2:
     *
     * ∫T_n(x)dx =
     *
     * T_{n+1}(x) / (2(n+1))
     * -
     * T_{n-1}(x) / (2(n-1))
     */
    public static double integralT(int n, double x) {
        validateDegree(n);
        validateX(x);

        if (n == 0) {
            return x;
        }

        if (n == 1) {
            return x * x / 2.0;
        }

        return T(n + 1, x)
                / (2.0 * (n + 1))
                -
                T(n - 1, x)
                        / (2.0 * (n - 1));
    }

    /**
     * Definite integral:
     *
     * ∫[a,b] T_n(x) dx
     */
    public static double definiteIntegralT(
            int n,
            double a,
            double b
    ) {
        validateDegree(n);
        validateX(a);
        validateX(b);

        return integralT(n, b)
                - integralT(n, a);
    }

    /**
     * Validate polynomial degree.
     */
    private static void validateDegree(int n) {
        if (n < 0) {
            throw new IllegalArgumentException(
                    "Degree n must be non-negative"
            );
        }
    }

    /**
     * Validate x ∈ [-1, 1].
     */
    private static void validateX(double x) {
        if (Double.isNaN(x) || Double.isInfinite(x)) {
            throw new IllegalArgumentException(
                    "x must be a finite number"
            );
        }

        if (x < -1.0 || x > 1.0) {
            throw new IllegalArgumentException(
                    "x must be in [-1, 1]"
            );
        }
    }

    /**
     * Validate coefficient array.
     */
    private static void validateCoefficients(
            double[] coefficients
    ) {
        if (coefficients == null
                || coefficients.length == 0) {

            throw new IllegalArgumentException(
                    "Coefficients must not be null or empty"
            );
        }
    }
}