#include <ceres/ceres.h>
#include <ceres/rotation.h>

using namespace std;

/*
 * We have a collection of points, potentially noisy.
 * We want to fit a curve through those points minimizing acceleration and
 * removing outliers.
 *
 * Observations: p(t)
 * Unknowns: x(t)
 *
 * Minimize:  lambda * || x(t-1) - 2 * x(t) + x(t+1) || + || p(t) - x(t) ||
 */
class GpsProblem {
 public:
  ~GpsProblem() {
    delete[] observations_;
    delete[] parameters_;
  }
  int numObservations() const { return numObservations_; }

  //void loadDataOrDie(const char* filename);
  void synth(int size);

  const double *observations() const { return observations_; }
  double *pointForObservation(int i) { return parameters_ + (i * 2); }

  double eval() const;

 private:
  int numObservations_;
  double* observations_;
  double* parameters_;
  double* truth_;
};

double lambda = 1.0;

struct GpsFitError {
  GpsFitError(double x, double y) {
    observed[0] = x;
    observed[1] = y;
  }

  template <typename T>
  bool operator()(const T* const point,
                  T* residuals) const {
    for (int i = 0; i < 2; ++i) {
      residuals[i] = point[i] - T(observed[i]);
    }
    return true;
  }

  static ceres::CostFunction* Create(double observedX,
                                     double observedY) {
    return (new ceres::AutoDiffCostFunction<GpsFitError, 2, 2>
            (new GpsFitError(observedX, observedY)));
  }

  double observed[2];
};

struct SmoothError {
  template <typename T>
  bool operator()(const T* const point1,
                  const T* const point2,
                  const T* const point3,
                  T* residuals) const {
    for (int i = 0; i < 2; ++i) {
      residuals[i] = T(lambda) * (point1[i] - T(2.0) * point2[i] + point3[i]);
    }
    return true;
  }

  static ceres::CostFunction* Create() {
    return (new ceres::AutoDiffCostFunction<SmoothError, 2, 2, 2, 2>
            (new SmoothError()));
  }
};


int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);

  /*
  if (argc != 2) {
    std::cerr << "usage: " << argv[0] <<" <noisy gps data>\n";
    return 1;
  }
  */

  ceres::Problem problem;
  GpsProblem gpsData;
  gpsData.synth(60 * 60 * 24);

  for (int i = 0; i < gpsData.numObservations(); ++i) {
    problem.AddResidualBlock(
        GpsFitError::Create(
            gpsData.observations()[2 * i + 0],
            gpsData.observations()[2 * i + 1]),
        new ceres::HuberLoss(10.0),
        //nullptr,
        gpsData.pointForObservation(i));
  }

  for (int i = 1; i < gpsData.numObservations() - 1; ++i) {
    problem.AddResidualBlock(
        SmoothError::Create(),
        nullptr, // no loss function
        gpsData.pointForObservation(i - 1),
        gpsData.pointForObservation(i + 0),
        gpsData.pointForObservation(i + 1));
  }

  double initial = gpsData.eval();

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  ceres::Solver::Summary summary;
  ceres::Solve(options, &problem, &summary);
  std::cout << summary.FullReport() << "\n";

  double after = gpsData.eval();
  cout << "Initial: " << initial << " final: " << after << endl;
  return 0;
}

double rnd(double min, double max) {
  return (max - min) * (double(rand()) / double(RAND_MAX)) + min;
}

void GpsProblem::synth(int size) {
  numObservations_ = size;
  observations_ = new double[size * 2];
  parameters_ = new double[size * 2];
  truth_ = new double[size * 2];

  for (int i = 0; i < size; ++i) {
    double angle = double(i) / size * 2 * M_PI;
    truth_[i * 2 + 0] = cos(angle);
    truth_[i * 2 + 1] = sin(angle);

    if (rnd(0, 1) < .1) {
      // outlier
      observations_[i * 2 + 0] = rnd(-10000, 10000);
      observations_[i * 2 + 1] = rnd(-10000, 10000);
    } else {
      // inlier
      observations_[i * 2 + 0] = truth_[i * 2 + 0] + rnd(-10, 10);
      observations_[i * 2 + 1] = truth_[i * 2 + 1] + rnd(-10, 10);
    }

  }
  for (int i = 0; i <size * 2; ++i) {
    parameters_[i] = observations_[i];
  }
}

double GpsProblem::eval() const {
  double sum = 0;
  for (int i = 0; i < numObservations_; ++i) {
    double d[2] = {
      truth_[i * 2 + 0] - parameters_[i * 2 + 0],
      truth_[i * 2 + 1] - parameters_[i * 2 + 1]
    };
    sum += sqrt(d[0] * d[0] + d[1] * d[1]);
  }
  return sum / numObservations_;
}
