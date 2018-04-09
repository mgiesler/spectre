// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares function for solving cubic equations

#pragma once

#include <cstddef>
#include <vector>

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Returns the real roots of a cubic equation ax^3 + bx^2 + cx + d = 0
 * \returns The real roots of a cubic equation.
 */

std::vector<double> real_cubic_roots(double a, double b, double c, double d);

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Provides a means to obtain a cubic root at higher precision
 * than real_cubic_roots, that utilizes the gsl cubic solver, which does not
 * allow for a specifiable precision.
 */

double numerical_real_root(double a, double b, double c, double d,
                           double initial_guess, size_t digits = 14);
