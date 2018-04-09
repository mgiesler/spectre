// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <vector>

#include "NumericalAlgorithms/RootFinding/CubicEquation.hpp"

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.CubicEquation.Precise",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  auto roots2 = real_cubic_roots(-1.0e-20, 0.0, 1.01, -1.0);
  // this check fails using gsl, as the rootfind is only approximate
  // CHECK(approx(0.99009900990099009) == roots2[1]);
  // numerically finding the root siginificantly improves the accuracy
  CHECK(approx(0.99009900990099009) ==
        numerical_real_root(-1.0e-20, 0.0, 1.01, -1.0, roots2[1], 14));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.CubicEquation.DeltaGTzero",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  // test case for reaching `(delta > 0)`, i.e. initial guess too large
  CHECK(approx(0.99009900990099009) ==
        numerical_real_root(-1.0e-20, 0.0, 1.01, -1.0, 1.0, 14));
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.CubicEquation.ThreeReal",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  // test handling of three real roots
  auto roots = real_cubic_roots(1.0, 2.0, -1.0, -2.0);
  CHECK(approx(-2.0) == roots[0]);
  CHECK(approx(-1.0) == roots[1]);
  CHECK(approx(1.0) == roots[2]);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.CubicEquation.a0",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  // test the reduction of a cubic equation to quadratic, a=0
  auto roots = real_cubic_roots(0.0, 1.5, 9.4, -0.001);
  CHECK(approx(-6.266773047839493) == roots[0]);
  CHECK(approx(0.00010638117282650023) == roots[1]);
}

SPECTRE_TEST_CASE("Unit.Numerical.RootFinding.CubicEquation.d0",
                  "[NumericalAlgorithms][RootFinding][Unit]") {
  // test the reduction of a cubic equation to quadratic, d=0
  auto roots = real_cubic_roots(1.0, 2.0, -1.0, 0.0);
  CHECK(approx(-2.414213562373095) == roots[0]);
  CHECK(approx(0.0) == roots[1]);
  CHECK(approx(0.4142135623730951) == roots[2]);
}
