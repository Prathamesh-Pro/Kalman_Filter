#include <iostream>
#include <iomanip>
#include <vector>
#include <random>
#include <array>

// ============================================================================
//  PART 1: 1D Kalman filter
//  State is ONE number (e.g. a temperature). Everything is a plain double.
// ============================================================================
class KalmanFilter1D {
public:
    KalmanFilter1D(double initial_estimate, double initial_error,
                   double process_noise, double measurement_noise)
        : estimate(initial_estimate),
          error_estimate(initial_error),
          Q(process_noise),      // how much we trust the process model
          R(measurement_noise) {} // how much we trust the sensor

    double update(double measurement) {
        // --- Prediction step ---
        // (no motion model here, so estimate stays the same,
        // but uncertainty grows because time passed)
        error_estimate += Q;

        // --- Update step ---
        double kalman_gain = error_estimate / (error_estimate + R);
        estimate = estimate + kalman_gain * (measurement - estimate);
        error_estimate = (1 - kalman_gain) * error_estimate;

        return estimate;
    }

    double getError() const { return error_estimate; }

private:
    double estimate;        // current best guess
    double error_estimate;  // uncertainty in that guess
    double Q;                // process noise covariance
    double R;                // measurement noise covariance
};

// ============================================================================
//  PART 2: 2D Kalman filter
//  State is TWO numbers: [position, velocity]. The sensor only measures
//  position; velocity is *inferred* by the filter. Every scalar from the 1D
//  version becomes a vector or matrix:
//
//      1D (scalar)                 2D (matrix)          meaning
//      ---------------------------------------------------------------------
//      estimate        (double) -> x  (2x1 vector)      [position, velocity]
//      error_estimate  (double) -> P  (2x2 matrix)      covariance of x
//      Q               (double) -> Q  (2x2 matrix)      process noise
//      R               (double) -> R  (double)          still scalar: 1 sensor
//      (none)                   -> F  (2x2 matrix)      motion model: pos += vel*dt
//      (none)                   -> H  (1x2 row)         which part of x we measure
//
//      P + Q                    -> F P F^T + Q
//      P / (P + R)              -> P H^T / (H P H^T + R)
//      est + K (z - est)        -> x + K (z - H x)
//      (1 - K) P                -> (I - K H) P
// ============================================================================

// --- tiny 2x2 linear algebra so we don't need a library like Eigen ---------
using Vec2 = std::array<double, 2>;
using Mat2 = std::array<std::array<double, 2>, 2>;

static Mat2 identity() { return {{{1, 0}, {0, 1}}}; }

static Mat2 transpose(const Mat2& A) {
    return {{{A[0][0], A[1][0]},
             {A[0][1], A[1][1]}}};
}

static Mat2 add(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C[i][j] = A[i][j] + B[i][j];
    return C;
}

static Mat2 sub(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C[i][j] = A[i][j] - B[i][j];
    return C;
}

static Mat2 mul(const Mat2& A, const Mat2& B) {
    Mat2 C{};
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < 2; ++j)
            C[i][j] = A[i][0] * B[0][j] + A[i][1] * B[1][j];
    return C;
}

static Vec2 mul(const Mat2& A, const Vec2& v) {
    return {A[0][0] * v[0] + A[0][1] * v[1],
            A[1][0] * v[0] + A[1][1] * v[1]};
}

static double dot(const Vec2& a, const Vec2& b) {
    return a[0] * b[0] + a[1] * b[1];
}

// column a (2x1) times row b (1x2) -> 2x2
static Mat2 outer(const Vec2& a, const Vec2& b) {
    return {{{a[0] * b[0], a[0] * b[1]},
             {a[1] * b[0], a[1] * b[1]}}};
}

class KalmanFilter2D {
public:
    KalmanFilter2D(Vec2 initial_estimate, Mat2 initial_error,
                   Mat2 process_noise, double measurement_noise, double dt)
        : x(initial_estimate),
          P(initial_error),
          Q(process_noise),        // how much we trust the motion model
          R(measurement_noise)     // how much we trust the sensor
    {
        // Motion model: position += velocity * dt ; velocity stays the same
        F = {{{1, dt},
              {0, 1}}};
        // Measurement model: the sensor sees position only  (z = 1*pos + 0*vel)
        H = {1, 0};
    }

    Vec2 update(double measurement) {
        // --- Prediction step ---
        // 1D:  estimate unchanged;   error_estimate += Q
        // 2D:  x = F x;              P = F P F^T + Q
        x = mul(F, x);
        P = add(mul(mul(F, P), transpose(F)), Q);

        // --- Update step ---
        // 1D:  K = P / (P + R)
        // 2D:  K = P H^T / (H P H^T + R)
        Vec2   PHt = mul(P, H);                 // P H^T          (2x1)
        double S   = dot(H, PHt) + R;           // H P H^T + R    (1x1 -> scalar)
        Vec2   K   = {PHt[0] / S, PHt[1] / S};  // Kalman gain    (2x1)

        // 1D:  estimate += K * (z - estimate)
        // 2D:  x        += K * (z - H x)
        double innovation = measurement - dot(H, x);
        x[0] += K[0] * innovation;   // correct position
        x[1] += K[1] * innovation;   // correct velocity (even though we never measured it!)

        // 1D:  P = (1 - K) P
        // 2D:  P = (I - K H) P
        P = mul(sub(identity(), outer(K, H)), P);

        return x;
    }

    const Mat2& getError() const { return P; }

private:
    Vec2   x;  // state estimate: [position, velocity]
    Mat2   P;  // uncertainty of the state (covariance matrix)
    Mat2   F;  // state transition (motion model)
    Vec2   H;  // measurement model (1x2 row vector)
    Mat2   Q;  // process noise covariance
    double R;  // measurement noise covariance (scalar: one sensor)
};

// ============================================================================
//  Demo: run both filters
// ============================================================================
int main() {
    std::cout << std::fixed << std::setprecision(2);
    std::default_random_engine gen;

    // -------------------- 1D: estimate a constant value --------------------
    std::cout << "=== 1D Kalman filter: constant value = 50 ===\n";
    {
        double true_value = 50.0;
        std::normal_distribution<double> noise(0.0, 4.0); // sensor noise std dev = 4

        KalmanFilter1D kf(0.0 /*initial guess*/, 1.0 /*initial uncertainty*/,
                          0.01 /*process noise*/, 4.0 /*measurement noise*/);

        for (int i = 0; i < 20; ++i) {
            double measurement = true_value + noise(gen);
            double filtered = kf.update(measurement);

            std::cout << "Step " << std::setw(2) << i
                      << " | Measured: " << std::setw(6) << measurement
                      << " | Filtered: " << std::setw(6) << filtered
                      << " | Error: " << kf.getError() << "\n";
        }
    }

    // ---------- 2D: track a moving object (position + velocity) ------------
    std::cout << "\n=== 2D Kalman filter: object at pos 10 moving at 2.0 m/s ===\n";
    {
        double dt            = 1.0;   // seconds between measurements
        double true_position = 10.0;
        double true_velocity = 2.0;
        std::normal_distribution<double> noise(0.0, 2.0); // sensor noise std dev = 2

        KalmanFilter2D kf({0.0, 0.0},                 // initial guess: [pos, vel]
                          {{{100, 0}, {0, 100}}},     // initial uncertainty (large: we know nothing)
                          {{{0.01, 0}, {0, 0.01}}},   // process noise
                          4.0,                        // measurement noise (std dev^2 = 2^2)
                          dt);

        for (int i = 0; i < 20; ++i) {
            double measurement = true_position + noise(gen);
            Vec2 filtered = kf.update(measurement);

            std::cout << "Step " << std::setw(2) << i
                      << " | True pos: " << std::setw(6) << true_position
                      << " | Measured: " << std::setw(6) << measurement
                      << " | Filtered pos: " << std::setw(6) << filtered[0]
                      << " | Filtered vel: " << std::setw(5) << filtered[1]
                      << " | Pos error: " << kf.getError()[0][0] << "\n";

            true_position += true_velocity * dt;   // the object moves before the next measurement
        }
    }

    return 0;
}
