// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <benchmark/benchmark.h>
#include <vector>

// #include "DataStructures/DataBox/DataBoxTag.hpp"
// #include "DataStructures/DataVector.hpp"
// #include "DataStructures/Index.hpp"
// #include "DataStructures/Tensor/Tensor.hpp"
// #include "DataStructures/Variables.hpp"
// #include "Domain/CoordinateMaps/Affine.hpp"
// #include "Domain/CoordinateMaps/CoordinateMap.hpp"
// #include "Domain/CoordinateMaps/ProductMaps.hpp"
// #include "Domain/Element.hpp"
// #include "Domain/LogicalCoordinates.hpp"
// #include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
// #include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"
// #include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include <array>
#include <cstddef>
#include <functional>
#include <limits>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <string>
#include <unordered_map>

#include "ControlSystem/FunctionOfTime.hpp"
#include "ControlSystem/PiecewisePolynomial.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/CoordinateMaps/CubicScale.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/CubicEquation.hpp"
#include "NumericalAlgorithms/RootFinding/NewtonRaphson.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

// Charm looks for this function but since we build without a main function or
// main module we just have it be empty
extern "C" void CkRegisterMainModule(void) {}

// This file is an example of how to do microbenchmark with Google Benchmark
// https://github.com/google/benchmark
// For two examples in different anonymous namespaces

namespace {
// Benchmark of push_back() in std::vector, following Chandler Carruth's talk
// at CppCon in 2015,
// https://www.youtube.com/watch?v=nXaxk27zwlk

// void bench_create(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    benchmark::DoNotOptimize(&v);
//    static_cast<void>(v);
//  }
// }
// BENCHMARK(bench_create);

// void bench_reserve(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//  }
// }
// BENCHMARK(bench_reserve);

// void bench_push_back(benchmark::State &state) {
//  while (state.KeepRunning()) {
//    std::vector<int> v;
//    v.reserve(1);
//    benchmark::DoNotOptimize(v.data());
//    v.push_back(42);
//    benchmark::ClobberMemory();
//  }
// }
// BENCHMARK(bench_push_back);
}  // namespace

namespace {
// In this anonymous namespace is an example of microbenchmarking the
// all_gradient routine for the GH system

// template <size_t Dim>
// struct Kappa : db::DataBoxTag {
//   using type = tnsr::abb<DataVector, Dim, Frame::Grid>;
//   static constexpr db::DataBoxString label = "Kappa";
// };
// template <size_t Dim>
// struct Psi : db::DataBoxTag {
//   using type = tnsr::aa<DataVector, Dim, Frame::Grid>;
//   static constexpr db::DataBoxString label = "Psi";
// };

// clang-tidy: don't pass be non-const reference
void bench_all_gradient(benchmark::State& state) {  // NOLINT
  constexpr size_t deriv_order = 2;
  const double t = 4.2;
  const double outer_b = 20.0;

  const CoordMapsTimeDependent::CubicScale scale_map(outer_b);

  const std::array<DataVector, deriv_order + 1> init_func_a{
      {{1.0}, {0.0007}, {-0.004}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t_A(t, init_func_a);
  FunctionOfTime& f_of_t_a = f_of_t_A;
  const std::array<DataVector, deriv_order + 1> init_func_b{
      {{1.0}, {-0.001}, {0.003}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> f_of_t_B(t, init_func_b);
  FunctionOfTime& f_of_t_b = f_of_t_B;

  const std::unordered_map<std::string, FunctionOfTime&> f_of_t_list = {
      {"expansion_a", f_of_t_a}, {"expansion_b", f_of_t_b}};

  const double a = f_of_t_a.func_and_deriv(t)[0][0];
  const double b = f_of_t_b.func_and_deriv(t)[0][0];

  while (state.KeepRunning()) {
    for (double j = 0.0; j < 20.0; j += 0.1) {
      const std::array<double, 1> point_xi{{j}};
      const std::array<double, 1> mapped_point{
          {point_xi[0] * (a + (b - a) * square(point_xi[0] / outer_b))}};
      benchmark::DoNotOptimize(scale_map.inverse(mapped_point, t, f_of_t_list));
    }
  }
}

// if google benchmark complains with: '***WARNING*** CPU scaling is enabled,
// the benchmark real time measurements may be noisy and will incur extra
//  overhead.'
//  this will fix it:
//  sudo cpupower frequency-set --governor performance
//  this will put if back to normal:
//  sudo cpupower frequency-set --governor powersave

BENCHMARK(bench_all_gradient)->Repetitions(50)->ReportAggregatesOnly();
// BENCHMARK(bench_all_gradient);
}  // namespace

BENCHMARK_MAIN()
