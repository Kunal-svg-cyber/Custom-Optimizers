/* Compiled reference kernels for the latency study.
 *
 * - Jacobi-preconditioned SVRG inner step and snapshot (the update rule of the PyTorch/JAX engines in
 *   their frozen-preconditioner mode with beta1 = 0), in double precision;
 * - recursive least squares (RLS), the standard per-tick online estimator, as the comparison point;
 * - a dot product (the signal-inference path of a linear model) and a plain SGD step.
 *
 * Plain portable C, single thread, no SIMD intrinsics (the compiler may vectorise with -O3). This measures
 * the cost of the ALGORITHM in compiled code; it is not a production low-latency component (no pinned
 * threads, lock-free queues, fixed-point arithmetic or kernel bypass).
 */
#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ns(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec * 1e9 + (double)t.tv_nsec;
}

/* mean-loss gradient of 0.5 * mean (x.w - y)^2 over all n rows */
void full_gradient(const double *X, const double *y, const double *w, int n, int d, double *out) {
    for (int j = 0; j < d; j++) out[j] = 0.0;
    for (int i = 0; i < n; i++) {
        const double *xi = X + (size_t)i * d;
        double r = -y[i];
        for (int j = 0; j < d; j++) r += xi[j] * w[j];
        for (int j = 0; j < d; j++) out[j] += xi[j] * r;
    }
    const double inv = 1.0 / (double)n;
    for (int j = 0; j < d; j++) out[j] *= inv;
}

static void batch_gradient(const double *X, const double *y, const double *w, const int *idx, int b, int d,
                           double *out) {
    for (int j = 0; j < d; j++) out[j] = 0.0;
    for (int k = 0; k < b; k++) {
        const double *xi = X + (size_t)idx[k] * d;
        double r = -y[idx[k]];
        for (int j = 0; j < d; j++) r += xi[j] * w[j];
        for (int j = 0; j < d; j++) out[j] += xi[j] * r;
    }
    const double inv = 1.0 / (double)b;
    for (int j = 0; j < d; j++) out[j] *= inv;
}

/* K inner SVRG steps with a FIXED snapshot: w <- w - lr * (g(w) - g(w_snap) + mu) / D.  idx is K x b. */
void svrg_steps(const double *X, const double *y, const double *inv_diag, const double *w_snap, const double *mu,
                int d, const int *idx, int K, int b, double lr, double *w) {
    double *gl = (double *)malloc(sizeof(double) * (size_t)d);
    double *gs = (double *)malloc(sizeof(double) * (size_t)d);
    for (int k = 0; k < K; k++) {
        const int *rows = idx + (size_t)k * b;
        batch_gradient(X, y, w, rows, b, d, gl);
        batch_gradient(X, y, w_snap, rows, b, d, gs);
        for (int j = 0; j < d; j++) w[j] -= lr * (gl[j] - gs[j] + mu[j]) * inv_diag[j];
    }
    free(gl);
    free(gs);
}

/* Full schedule: snapshot at step 0 and every `interval` steps, as in the engines and the NumPy oracle. */
void svrg_run(const double *X, const double *y, const double *inv_diag, int n, int d, const int *idx, int K, int b,
              int interval, double lr, double *w) {
    double *w_snap = (double *)malloc(sizeof(double) * (size_t)d);
    double *mu = (double *)malloc(sizeof(double) * (size_t)d);
    int done = 0;
    while (done < K) {
        memcpy(w_snap, w, sizeof(double) * (size_t)d);
        full_gradient(X, y, w, n, d, mu);
        int chunk = (K - done < interval) ? (K - done) : interval;
        svrg_steps(X, y, inv_diag, w_snap, mu, d, idx + (size_t)done * b, chunk, b, lr, w);
        done += chunk;
    }
    free(w_snap);
    free(mu);
}

/* ---- timing helpers: each returns nanoseconds per operation (best of `trials`) ---- */

double time_svrg_step_ns(const double *X, const double *y, const double *inv_diag, int n, int d, const int *idx,
                         int K, int b, double lr, int trials) {
    double *w = (double *)calloc((size_t)d, sizeof(double));
    double *w_snap = (double *)calloc((size_t)d, sizeof(double));
    double *mu = (double *)malloc(sizeof(double) * (size_t)d);
    full_gradient(X, y, w, n, d, mu);
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        memset(w, 0, sizeof(double) * (size_t)d);
        double t0 = now_ns();
        svrg_steps(X, y, inv_diag, w_snap, mu, d, idx, K, b, lr * 1e-3, w);
        double dt = (now_ns() - t0) / (double)K;
        if (dt < best) best = dt;
    }
    free(w); free(w_snap); free(mu);
    return best;
}

double time_snapshot_ns(const double *X, const double *y, int n, int d, int trials) {
    double *w = (double *)calloc((size_t)d, sizeof(double));
    double *mu = (double *)malloc(sizeof(double) * (size_t)d);
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        full_gradient(X, y, w, n, d, mu);
        double dt = now_ns() - t0;
        if (dt < best) best = dt;
    }
    free(w); free(mu);
    return best;
}

/* plain SGD step on a batch (no variance reduction) */
double time_sgd_step_ns(const double *X, const double *y, int d, const int *idx, int K, int b, int trials) {
    double *w = (double *)calloc((size_t)d, sizeof(double));
    double *g = (double *)malloc(sizeof(double) * (size_t)d);
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        memset(w, 0, sizeof(double) * (size_t)d);
        double t0 = now_ns();
        for (int k = 0; k < K; k++) {
            batch_gradient(X, y, w, idx + (size_t)k * b, b, d, g);
            for (int j = 0; j < d; j++) w[j] -= 1e-6 * g[j];
        }
        double dt = (now_ns() - t0) / (double)K;
        if (dt < best) best = dt;
    }
    free(w); free(g);
    return best;
}

/* recursive least squares update with forgetting factor lam: O(d^2) per observation */
double time_rls_update_ns(const double *X, const double *y, int d, int K, double lam, int trials) {
    double *P = (double *)malloc(sizeof(double) * (size_t)d * d);
    double *w = (double *)calloc((size_t)d, sizeof(double));
    double *Px = (double *)malloc(sizeof(double) * (size_t)d);
    double best = 1e300;
    for (int t = 0; t < trials; t++) {
        for (int i = 0; i < d; i++)
            for (int j = 0; j < d; j++) P[(size_t)i * d + j] = (i == j) ? 1e3 : 0.0;
        memset(w, 0, sizeof(double) * (size_t)d);
        double t0 = now_ns();
        for (int k = 0; k < K; k++) {
            const double *x = X + (size_t)k * d;
            double xPx = 0.0, pred = 0.0;
            for (int i = 0; i < d; i++) {
                double s = 0.0;
                const double *Pi = P + (size_t)i * d;
                for (int j = 0; j < d; j++) s += Pi[j] * x[j];
                Px[i] = s;
                xPx += x[i] * s;
                pred += w[i] * x[i];
            }
            const double inv = 1.0 / (lam + xPx);
            const double err = y[k] - pred;
            for (int i = 0; i < d; i++) w[i] += Px[i] * inv * err;
            for (int i = 0; i < d; i++) {
                double *Pi = P + (size_t)i * d;
                const double ki = Px[i] * inv;
                for (int j = 0; j < d; j++) Pi[j] = (Pi[j] - ki * Px[j]) / lam;
            }
        }
        double dt = (now_ns() - t0) / (double)K;
        if (dt < best) best = dt;
    }
    free(P); free(w); free(Px);
    return best;
}

/* signal inference for a linear model: one dot product per row */
double time_dot_ns(const double *X, const double *w, int d, int K, int trials, double *sink) {
    double best = 1e300, acc = 0.0;
    for (int t = 0; t < trials; t++) {
        double t0 = now_ns();
        for (int k = 0; k < K; k++) {
            const double *x = X + (size_t)k * d;
            double s = 0.0;
            for (int j = 0; j < d; j++) s += x[j] * w[j];
            acc += s;
        }
        double dt = (now_ns() - t0) / (double)K;
        if (dt < best) best = dt;
    }
    *sink = acc;
    return best;
}
