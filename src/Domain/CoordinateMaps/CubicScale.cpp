// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/CubicScale.hpp"

#include <array>
#include <functional>
#include <ostream>
#include <pup.h>
#include <pup_stl.h>
#include <vector>

#include "ControlSystem/FunctionOfTime.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/RootFinding/CubicEquation.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace CoordMapsTimeDependent {

CubicScale::CubicScale(const double outer_boundary)
    : outer_boundary_(outer_boundary) {}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> CubicScale::operator()(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  return {{source_coords[0] *
           (a_of_t +
            (b_of_t - a_of_t) * square(source_coords[0] / outer_boundary_))}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> CubicScale::inverse(
    const std::array<T, 1>& target_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  // the original coordinates are found by solving for the roots
  // of (b-a)/X^2*\xi^3 + a*\xi - x = 0, where a and b are the FunctionsOfTime,
  // X is the outer_boundary, and x represents the mapped coordinates
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  const double cubic_coef_a = (b_of_t - a_of_t) / square(outer_boundary_);
  const auto roots =
      real_cubic_roots(cubic_coef_a, 0.0, a_of_t, -target_coords[0]);

  // The map is invertible for all r<=outer_boundary as long as 0<a<3b/2.
  if (b_of_t <= 2.0 / 3.0 * a_of_t) {
    ERROR("The map is only invertible if expansion_b < expansion_a*2/3, however"
          << " expansion_b = " << b_of_t << " and expansion_a = " << a_of_t
          << " does not satisfy this criterion.");
  }

  std::vector<double> good_roots;
  if (roots.size() > 1) {
    // Here we need to find the appropriate root:
    // First, we rule out negative solutions
    // Second, we rule out those with negative slope, since the mapping
    // should be strictly increasing
    for (auto root : roots) {
      const double root_deriv = a_of_t + 3.0 * cubic_coef_a * square(root);
      if (root_deriv >= 0.0 and root > 0.0) {
        good_roots.emplace_back(root);
      }
    }
    if (good_roots.size() != 1) {
      ERROR("Found " << good_roots.size()
                     << " roots. A unique solution is required by this map.");
    }
    return {{numerical_real_root(cubic_coef_a, 0.0, a_of_t, -target_coords[0],
                                 good_roots[0])}};
  }

  return {{numerical_real_root(cubic_coef_a, 0.0, a_of_t, -target_coords[0],
                               roots[0])}};
}

template <>
std::array<tt::remove_cvref_wrap_t<DataVector>, 1> CubicScale::inverse(
    const std::array<DataVector, 1>& target_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  DataVector solutions{target_coords[0]};
  for (auto& solution : solutions) {
    solution = inverse<double>({{solution}}, time, map_list)[0];
  }
  return {{solutions}};
}

template <>
std::array<tt::remove_cvref_wrap_t<std::reference_wrapper<const DataVector>>, 1>
CubicScale::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 1>&
        target_coords,
    const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  DataVector solutions{target_coords[0]};
  for (auto& solution : solutions) {
    solution = inverse<double>({{solution}}, time, map_list)[0];
  }
  return {{solutions}};
}

template <typename T>
std::array<tt::remove_cvref_wrap_t<T>, 1> CubicScale::frame_velocity(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto dt_a_of_t = map_list.at(f_of_t_a_).func_and_deriv(time)[1][0];
  const auto dt_b_of_t = map_list.at(f_of_t_b_).func_and_deriv(time)[1][0];
  const auto frame_vel =
      source_coords[0] *
      (dt_a_of_t +
       (dt_b_of_t - dt_a_of_t) * square(source_coords[0] / outer_boundary_));

  return {{frame_vel}};
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> CubicScale::jacobian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];
  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> jac{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};

  get<0, 0>(jac) =
      a_of_t +
      3.0 * (b_of_t - a_of_t) * square(source_coords[0] / outer_boundary_);

  return jac;
}

template <typename T>
tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>
CubicScale::inv_jacobian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t = map_list.at(f_of_t_a_).func(time)[0][0];
  const auto b_of_t = map_list.at(f_of_t_b_).func(time)[0][0];

  tnsr::Ij<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> inv_jac{
      make_with_value<tt::remove_cvref_wrap_t<T>>(
          dereference_wrapper(source_coords[0]), 0.0)};

  get<0, 0>(inv_jac) = 1.0 / (a_of_t +
                              3.0 * (b_of_t - a_of_t) *
                                  square(source_coords[0] / outer_boundary_));

  return inv_jac;
}

template <typename T>
tnsr::Iaa<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame> CubicScale::hessian(
    const std::array<T, 1>& source_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept {
  const auto a_of_t_and_derivs = map_list.at(f_of_t_a_).func_and_2_derivs(time);
  const auto b_of_t_and_derivs = map_list.at(f_of_t_b_).func_and_2_derivs(time);

  auto result{
      make_with_value<tnsr::Iaa<tt::remove_cvref_wrap_t<T>, 1, Frame::NoFrame>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  // time-time
  get<0, 0, 0>(result) = a_of_t_and_derivs[2][0] * source_coords[0] +
                         (b_of_t_and_derivs[2][0] - a_of_t_and_derivs[2][0]) *
                             cube(source_coords[0]) / square(outer_boundary_);
  // time-space
  get<0, 0, 1>(result) =
      a_of_t_and_derivs[1][0] +
      3.0 * (b_of_t_and_derivs[1][0] - a_of_t_and_derivs[1][0]) *
          square(source_coords[0] / outer_boundary_);
  // space-space
  get<0, 1, 1>(result) = 6.0 *
                         (b_of_t_and_derivs[0][0] - a_of_t_and_derivs[0][0]) *
                         source_coords[0] / square(outer_boundary_);

  return result;
}

void CubicScale::pup(PUP::er& p) noexcept {
  p | f_of_t_a_;
  p | f_of_t_b_;
  p | outer_boundary_;
}

bool operator==(const CoordMapsTimeDependent::CubicScale& lhs,
                const CoordMapsTimeDependent::CubicScale& rhs) noexcept {
  return lhs.f_of_t_a_ == rhs.f_of_t_a_ and lhs.f_of_t_b_ == rhs.f_of_t_b_ and
         lhs.outer_boundary_ == rhs.outer_boundary_;
}

// Explicit instantiations
/// \cond
template std::array<tt::remove_cvref_wrap_t<double>, 1> CubicScale::inverse(
    const std::array<double, 1>& target_coords, const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept;
template std::array<
    tt::remove_cvref_wrap_t<std::reference_wrapper<const double>>, 1>
CubicScale::inverse(
    const std::array<std::reference_wrapper<const double>, 1>& target_coords,
    const double time,
    const std::unordered_map<std::string, FunctionOfTime&>& map_list) const
    noexcept;

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1> CubicScale::    \
  operator()(const std::array<DTYPE(data), 1>& source_coords,                  \
             const double time,                                                \
             const std::unordered_map<std::string, FunctionOfTime&>& map_list) \
      const noexcept;                                                          \
  template std::array<tt::remove_cvref_wrap_t<DTYPE(data)>, 1>                 \
  CubicScale::frame_velocity(                                                  \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  CubicScale::jacobian(                                                        \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Ij<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>   \
  CubicScale::inv_jacobian(                                                    \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;                                                          \
  template tnsr::Iaa<tt::remove_cvref_wrap_t<DTYPE(data)>, 1, Frame::NoFrame>  \
  CubicScale::hessian(                                                         \
      const std::array<DTYPE(data), 1>& source_coords, const double time,      \
      const std::unordered_map<std::string, FunctionOfTime&>& map_list)        \
      const noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector,
                                      std::reference_wrapper<const double>,
                                      std::reference_wrapper<const DataVector>))
#undef DTYPE
#undef INSTANTIATE
/// \endcond
}  // namespace CoordMapsTimeDependent
