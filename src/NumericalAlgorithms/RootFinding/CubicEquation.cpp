// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/RootFinding/CubicEquation.hpp"

#include <cmath>
#include <gsl/gsl_poly.h>
#include <limits>
#include <utility>

#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"

std::vector<double> real_cubic_roots(const double a, const double b,
                                     const double c, const double d) {
  std::vector<double> the_roots{std::numeric_limits<double>::signaling_NaN(),
                                std::numeric_limits<double>::signaling_NaN(),
                                std::numeric_limits<double>::signaling_NaN()};

  int num_real_roots = 0;
  if (a == 0.0) {
    // quadratic
    num_real_roots =
        gsl_poly_solve_quadratic(b, c, d, &the_roots[0], &the_roots[1]);
  } else {
    // legitimate cubic equation
    num_real_roots = gsl_poly_solve_cubic(b / a, c / a, d / a, &the_roots[0],
                                          &the_roots[1], &the_roots[2]);
  }

  the_roots.resize(static_cast<size_t>(num_real_roots));
  return the_roots;
}

double numerical_real_root(const double a, const double b, const double c,
                           const double d, const double initial_guess,
                           const size_t digits) {
  const double tolerance = 1.0 / std::pow(10, digits);
  const double error =
      d + initial_guess * (c + initial_guess * (b + a * initial_guess));

  if (fabs(error) < tolerance) {
    return initial_guess;
  }

  // estimate the interval for the root finder
  const double deriv = initial_guess * (3.0 * a * initial_guess + 2.0 * b) + c;
  // estimate delta ~ initial_guess - true_soln
  const double delta = error / deriv;

  // TEMPORARY NOTE:
  // adapted from SpEC (however, this is done differently there and
  // copying directly results in a failure to choose the correct interval.
  // However, BracketByExpanding gets called, which may provide the
  // correct interval. The modifications here provide, what I think is,
  // the correct interval)
  // double check, and fix the wrong implementation.
  double x1 = initial_guess;
  double x2 = initial_guess;
  if (delta < 0) {
    x2 -= 2 * delta;
  } else {
    x1 -= 2 * delta;
  }
  if (fabs(x1 - x2) < tolerance * initial_guess) {
    x1 -= tolerance * initial_guess;
    x2 += tolerance * initial_guess;
  }

  const auto f_lambda = [&a, &b, &c, &d ](double x) noexcept {
    return std::make_pair(x * (x * (a * x + b) + c) + d,
                          x * (3.0 * a * x + 2.0 * b) + c);
  };
  return RootFinder::newton_raphson(f_lambda, initial_guess, x1, x2, digits);
}
