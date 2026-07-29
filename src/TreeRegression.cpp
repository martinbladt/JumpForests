/*

Functions for regression trees

*/

#include "TreeRegression.h"

#include <cmath>
#include <cfloat>
#include <limits>

namespace {

long double safeRegressionSplitScore(long double score) {
  if (isnan(score)) {
    return -numeric_limits<long double>::infinity();
  }
  if (score <= 0) {
    return 0;
  }
  if (!isfinite(score)) {
    return numeric_limits<long double>::max();
  }
  return score;
}

void compensatedAdd(long double value, long double& sum, long double& correction) {
  long double updated = sum + value;
  if (abs(sum) >= abs(value)) {
    correction += (sum - updated) + value;
  } else {
    correction += (value - updated) + sum;
  }
  sum = updated;
}

bool isNonNegativeInteger(double value) {
  return value >= 0 && value == floor(value);
}

// log(1 + x) - x, evaluated without losing its quadratic leading term.
long double log1pMinusArgument(long double x) {
  if (abs(x) < 1e-4L) {
    long double power = x * x;
    long double result = -power / 2;
    for (unsigned int degree = 3; degree <= 12; ++degree) {
      power *= x;
      long double term = power / degree;
      result += degree % 2 == 0 ? -term : term;
    }
    return result;
  }
  return log1pl(x) - x;
}

// (log(1 + x) - x) / x^2, including its continuous value at zero.
long double log1pMinusArgumentOverSquare(long double x) {
  if (abs(x) < 1e-4L) {
    long double result = -0.5L;
    long double power = x;
    for (unsigned int degree = 3; degree <= 12; ++degree) {
      long double term = power / degree;
      result += degree % 2 == 0 ? -term : term;
      power *= x;
    }
    return result;
  }
  return (log1pl(x) / x - 1) / x;
}

// (exp(x) - 1) / x - 1, evaluated without cancelling its linear leading term.
long double expm1RatioMinusOne(long double x) {
  if (abs(x) < 1e-4L) {
    long double term = x / 2;
    long double result = term;
    for (unsigned int denominator = 3; denominator <= 12; ++denominator) {
      term *= x / denominator;
      result += term;
    }
    return result;
  }
  return expm1l(x) / x - 1;
}

long double expm1Ratio(long double x) {
  if (abs(x) < 1e-4L) {
    return 1 + expm1RatioMinusOne(x);
  }
  return expm1l(x) / x;
}

long double poissonPotentialFromDifference(long double difference, long double reference, long double mean) {
  if (reference == 0) {
    return 0;
  }
  long double ratio_difference = difference / reference;
  if (ratio_difference > -0.5L && ratio_difference < 1) {
    long double log_ratio = log1pl(ratio_difference);
    return reference *
      (log1pMinusArgument(ratio_difference) +
       ratio_difference * log_ratio);
  }
  if (mean == 0) {
    return reference;
  }
  return mean * (logl(mean) - logl(reference)) - difference;
}

long double meanFromCompensatedSum(long double sum, long double correction, size_t count) {
  long double denominator = static_cast<long double>(count);
  return sum / denominator + correction / denominator;
}

long double differenceOfCompensatedMeans(long double first_sum, long double first_correction, size_t first_count,
                                         long double second_sum, long double second_correction, size_t second_count) {
  long double difference = 0;
  long double correction = 0;
  compensatedAdd(first_sum / static_cast<long double>(first_count), difference, correction);
  compensatedAdd(-second_sum / static_cast<long double>(second_count), difference, correction);
  compensatedAdd(first_correction / static_cast<long double>(first_count), difference, correction);
  compensatedAdd(-second_correction / static_cast<long double>(second_count), difference, correction);
  return difference + correction;
}

long double absoluteLossDifference(long double response, long double parent_center, long double daughter_center) {
  long double lower_center = min(parent_center, daughter_center);
  long double upper_center = max(parent_center, daughter_center);
  if (response >= upper_center) {
    return daughter_center - parent_center;
  }
  if (response <= lower_center) {
    return parent_center - daughter_center;
  }
  return abs(response - parent_center) - abs(response - daughter_center);
}

long double integrateClippedResidual(long double response, long double from, long double to, long double delta) {
  if (from == to) {
    return 0;
  }
  if (to < from) {
    return -integrateClippedResidual(response, to, from, delta);
  }

  long double lower_break = response - delta;
  long double upper_break = response + delta;
  long double integral = 0;
  long double current = from;

  if (current < lower_break) {
    long double endpoint = min(to, lower_break);
    integral -= delta * (endpoint - current);
    current = endpoint;
  }
  if (current < to && current < upper_break) {
    long double endpoint = min(to, upper_break);
    long double start_residual = current - response;
    long double end_residual = endpoint - response;
    integral +=
      (endpoint - current) * (start_residual + end_residual) / 2;
    current = endpoint;
  }
  if (current < to) {
    integral += delta * (to - current);
  }
  return integral;
}

long double huberLossDifference(
    long double response, long double parent_center,
    long double daughter_center, long double delta) {
  // d rho(y-c) / dc = 2 * clamp(c-y, -delta, delta).
  return -2 * integrateClippedResidual(response, parent_center, daughter_center, delta);
}

long double signedLogSumMagnitude(int sign_first, long double log_first, int sign_second, long double log_second,
                                  int sign_third, long double log_third) {
  long double largest = max(log_first, max(log_second, log_third));
  if (largest == -numeric_limits<long double>::infinity()) {
    return largest;
  }
  long double scaled_sum = 0;
  if (log_first != -numeric_limits<long double>::infinity()) {
    scaled_sum += sign_first * expl(log_first - largest);
  }
  if (log_second != -numeric_limits<long double>::infinity()) {
    scaled_sum += sign_second * expl(log_second - largest);
  }
  if (log_third != -numeric_limits<long double>::infinity()) {
    scaled_sum += sign_third * expl(log_third - largest);
  }
  if (scaled_sum == 0) {
    return -numeric_limits<long double>::infinity();
  }
  return largest + logl(abs(scaled_sum));
}

long double logSumExp(long double first, long double second) {
  if (first == -numeric_limits<long double>::infinity()) {
    return second;
  }
  if (second == -numeric_limits<long double>::infinity()) {
    return first;
  }
  long double largest = max(first, second);
  return largest + log1pl(expl(min(first, second) - largest));
}

}

RegressionSplitRule regressionSplitRuleFromString(const string& splitrule) {
  if (splitrule == "mse" || splitrule == "variance") {
    return RegressionSplitRule::MSE;
  }
  if (splitrule == "mae") {
    return RegressionSplitRule::MAE;
  }
  if (splitrule == "binomial") {
    return RegressionSplitRule::Binomial;
  }
  if (splitrule == "negativebinomial") {
    return RegressionSplitRule::NegativeBinomial;
  }
  if (splitrule == "poisson") {
    return RegressionSplitRule::Poisson;
  }
  if (splitrule == "gamma") {
    return RegressionSplitRule::Gamma;
  }
  if (splitrule == "inversegaussian") {
    return RegressionSplitRule::InverseGaussian;
  }
  if (splitrule == "tweedie") {
    return RegressionSplitRule::Tweedie;
  }
  if (splitrule == "huber") {
    return RegressionSplitRule::Huber;
  }
  throw runtime_error("Invalid regression splitrule; choose mse, variance, mae, binomial, negativebinomial, poisson, gamma, inversegaussian, tweedie or huber");
}

bool regressionSplitRuleUsesParameter(RegressionSplitRule splitrule) {
  return splitrule == RegressionSplitRule::NegativeBinomial ||
    splitrule == RegressionSplitRule::Tweedie ||
    splitrule == RegressionSplitRule::Huber;
}

void validateRegressionResponse(const vector<double>& response, RegressionSplitRule splitrule, double splitrule_par) {
  for (double value : response) {
    if (!isfinite(value)) {
      throw runtime_error("Regression response values must be finite");
    }
  }

  switch (splitrule) {
    case RegressionSplitRule::Binomial:
      for (double value : response) {
        if (value < 0 || value > 1) {
          throw runtime_error(
            "The binomial splitrule requires response values in [0, 1]");
        }
      }
      break;
    case RegressionSplitRule::NegativeBinomial:
      for (double value : response) {
        if (!isNonNegativeInteger(value)) {
          throw runtime_error(
            "The negativebinomial splitrule requires non-negative integer responses");
        }
      }
      break;
    case RegressionSplitRule::Poisson:
      for (double value : response) {
        if (!isNonNegativeInteger(value)) {
          throw runtime_error(
            "The poisson splitrule requires non-negative integer responses");
        }
      }
      break;
    case RegressionSplitRule::Gamma:
      for (double value : response) {
        if (value <= 0) {
          throw runtime_error(
            "The gamma splitrule requires strictly positive responses");
        }
      }
      break;
    case RegressionSplitRule::InverseGaussian:
      for (double value : response) {
        if (value <= 0) {
          throw runtime_error(
            "The inversegaussian splitrule requires strictly positive responses");
        }
      }
      break;
    case RegressionSplitRule::Tweedie:
      if (splitrule_par == 0) {
        break;
      }
      if (splitrule_par == 1) {
        for (double value : response) {
          if (!isNonNegativeInteger(value)) {
            throw runtime_error(
              "The tweedie splitrule with xi = 1 requires non-negative integer responses");
          }
        }
      } else if (splitrule_par > 1 && splitrule_par < 2) {
        for (double value : response) {
          if (value < 0) {
            throw runtime_error(
              "The tweedie splitrule with 1 < xi < 2 requires non-negative responses");
          }
        }
      } else if (splitrule_par >= 2) {
        for (double value : response) {
          if (value <= 0) {
            throw runtime_error(
              "The tweedie splitrule with xi >= 2 requires strictly positive responses");
          }
        }
      }
      break;
    case RegressionSplitRule::MSE:
    case RegressionSplitRule::MAE:
    case RegressionSplitRule::Huber:
      break;
  }
}

// constructor for RegressionTree
//--------------------------------------------------------------------------------------

RegressionTree::RegressionTree(vector<size_t> subset_indices, vector<size_t> estimation_indices) {
  this->node_sizes.push_back(subset_indices.size());
  this->node_obs.push_back(std::move(subset_indices));
  this->holdout_node_obs.push_back(std::move(estimation_indices));
}

// functions for growing regression trees
//--------------------------------------------------------------------------------------

long double RegressionTree::computeSum(const vector<size_t>& indices) {
  long double sum = 0;
  long double correction = 0;
  for (size_t i : indices) {
    compensatedAdd(data->get_y(i), sum, correction);
  }
  return sum + correction;
}

long double RegressionTree::computeMedianSorted(
    const vector<double>& response_values) {
  if (response_values.empty()) {
    return 0;
  }

  size_t n = response_values.size();
  long double median = response_values[n / 2];
  if (n % 2 == 0) {
    median =
      (static_cast<long double>(response_values[n / 2 - 1]) + median) / 2;
  }
  return median;
}

long double RegressionTree::computeHuberCenterSorted(const vector<double>& response_values) {
  if (response_values.empty()) {
    return 0;
  }

  const size_t n = response_values.size();
  const long double delta = splitrule_par;
  long double response_range = static_cast<long double>(response_values.back()) - response_values.front();
  if (delta >= response_range) {
    long double sum = 0;
    long double correction = 0;
    for (double response : response_values) {
      compensatedAdd(response, sum, correction);
    }
    return (sum + correction) / n;
  }
  size_t lower_id = 0;
  size_t upper_id = 0;
  size_t active = 0;
  long double position = static_cast<long double>(response_values.front()) - delta;
  long double clipped_score = -static_cast<long double>(n) * delta;

  while (lower_id < n || upper_id < n) {
    long double next_lower = lower_id < n ?
      static_cast<long double>(response_values[lower_id]) - delta : numeric_limits<long double>::infinity();
    long double next_upper = upper_id < n ?
      static_cast<long double>(response_values[upper_id]) + delta : numeric_limits<long double>::infinity();
    long double next_position = min(next_lower, next_upper);

    if (next_position > position && active > 0) {
      long double next_score = clipped_score + static_cast<long double>(active) * (next_position - position);
      if (clipped_score <= 0 && next_score >= 0) {
        return position - clipped_score / static_cast<long double>(active);
      }
      clipped_score = next_score;
    }
    position = next_position;

    while (lower_id < n && static_cast<long double>(response_values[lower_id]) - delta == position) {
      ++active;
      ++lower_id;
    }
    while (upper_id < n && static_cast<long double>(response_values[upper_id]) + delta == position) {
      --active;
      ++upper_id;
    }

    if (clipped_score == 0) {
      return position;
    }
  }

  // The clipped score always crosses zero, so this is only a numerical fallback.
  return static_cast<long double>(response_values[n / 2]);
}

long double RegressionTree::computeMSESplitValue(size_t n_left, long double sum_left, size_t n_right, long double sum_right) {
  (void) sum_left;
  (void) sum_right;
  return safeRegressionSplitScore(n_left * candidate_left_difference * candidate_left_difference + n_right * candidate_right_difference * candidate_right_difference);
}

long double RegressionTree::computeMAESplitValue(
    const vector<double>& left_responses,
    const vector<double>& right_responses,
    long double parent_center) {
  long double left_center = computeMedianSorted(left_responses);
  long double right_center = computeMedianSorted(right_responses);
  long double score = 0;
  long double correction = 0;
  for (double response : left_responses) {
    compensatedAdd(absoluteLossDifference(response, parent_center, left_center), score, correction);
  }
  for (double response : right_responses) {
    compensatedAdd(absoluteLossDifference(response, parent_center, right_center), score, correction);
  }
  score += correction;
  return safeRegressionSplitScore(score);
}

long double RegressionTree::computeBinomialSplitValue(size_t n_left, long double sum_left, long double failure_sum_left,
                                                      size_t n_right, long double sum_right, long double failure_sum_right) {
  (void) sum_left;
  (void) sum_right;
  (void) failure_sum_left;
  (void) failure_sum_right;
  const long double mean_parent = candidate_parent_mean;
  const long double failure_mean_parent = candidate_failure_parent_mean;

  auto deviance = [mean_parent, failure_mean_parent](long double mean, long double difference, long double failure_mean, long double failure_difference) {
    // Writing Bernoulli KL as two Poisson potentials makes the cancellation
    // of their linear terms explicit. Success and failure sums are accumulated
    // independently so rare tail mass is not lost when a mean rounds to 0/1.
    return 2 * (poissonPotentialFromDifference(difference, mean_parent, mean) + poissonPotentialFromDifference(failure_difference, failure_mean_parent, failure_mean));
  };

  return safeRegressionSplitScore(n_left * deviance(candidate_left_mean, candidate_left_difference, candidate_left_failure_mean, candidate_left_failure_difference) +
                                  n_right * deviance(candidate_right_mean, candidate_right_difference, candidate_right_failure_mean, candidate_right_failure_difference));
}

long double RegressionTree::computeNegativeBinomialSplitValue(size_t n_left, long double sum_left, size_t n_right, long double sum_right) {
  (void) sum_left;
  (void) sum_right;
  const long double mean_parent = candidate_parent_mean;
  const long double size = splitrule_par;

  auto deviance = [mean_parent, size](
      long double mean, long double difference) {
    if (mean_parent == 0) {
      return static_cast<long double>(0);
    }
    if (mean == 0) {
      return 2 * size * log1pl(mean_parent / size);
    }

    long double ratio_difference = difference / mean_parent;
    long double ratio = 1 + ratio_difference;
    long double smallest_mean = min(mean, mean_parent);

    // The deviance tends to zero linearly in k. Directly subtracting its two
    // logarithmic terms therefore discards every useful bit for very small k.
    // Integrating its derivative at k = 0 gives a stable local expansion.
    if (size <= sqrtl(LDBL_EPSILON) * smallest_mean) {
      long double limit_coefficient =
        ratio_difference < -0.5L || ratio_difference > 1 ?
        ratio_difference - (logl(mean) - logl(mean_parent)) :
        -log1pMinusArgument(ratio_difference);
      long double leading = size * limit_coefficient;
      long double correction =
        size * size * ratio_difference * ratio_difference /
        (2 * mean);
      return 2 * max(leading - correction, static_cast<long double>(0));
    }

    // For large k, evaluate the exact tail from the Poisson limit as a
    // convergent series. Replacing the criterion by its limit can discard a
    // small but decision-relevant ordering correction.
    long double tail_scale = mean_parent + size;
    long double tail_ratio = difference / tail_scale;
    if (abs(tail_ratio) < 1e-4L) {
      long double power_term = tail_ratio * tail_ratio;
      long double tail_series = power_term / 2;
      for (unsigned int degree = 3; degree <= 12; ++degree) {
        power_term *= tail_ratio;
        long double term = power_term / (degree * static_cast<long double>(degree - 1));
        tail_series += degree % 2 == 0 ? term : -term;
      }
      long double potential = poissonPotentialFromDifference(difference, mean_parent, mean) - tail_scale * tail_series;
      return 2 * max(potential, static_cast<long double>(0));
    }

    if (!isfinite(ratio) || ratio < 0.5L || ratio > 2) {
      long double potential =
        mean * (logl(mean) - logl(mean_parent)) - (mean + size) * (logl(mean + size) - logl(mean_parent + size));
      return 2 * max(potential, static_cast<long double>(0));
    }

    // Factor out (mean / parent - 1)^2. This also retains precision when the
    // two means are close but k is not in the small-k regime above.
    long double scaled_size = size / mean_parent;
    if (!isfinite(scaled_size)) {
      return 2 * poissonPotentialFromDifference(difference, mean_parent, mean);
    }
    long double denominator = 1 + scaled_size;
    long double adjusted_difference = ratio_difference / denominator;
    long double coefficient = scaled_size / denominator + ratio * log1pMinusArgumentOverSquare(ratio_difference) - ((ratio + scaled_size) / denominator / denominator) *
                              log1pMinusArgumentOverSquare(adjusted_difference);
    long double potential = mean_parent * ratio_difference * ratio_difference * coefficient;
    return 2 * max(potential, static_cast<long double>(0));
  };

  return safeRegressionSplitScore(n_left * deviance(candidate_left_mean, candidate_left_difference) + n_right * deviance(candidate_right_mean, candidate_right_difference));
}

long double RegressionTree::computePoissonSplitValue(
    size_t n_left, long double sum_left, size_t n_right,
    long double sum_right) {
  (void) sum_left;
  (void) sum_right;

  return safeRegressionSplitScore(2 * n_left * poissonPotentialFromDifference(candidate_left_difference, candidate_parent_mean, candidate_left_mean) +
                                  2 * n_right * poissonPotentialFromDifference(candidate_right_difference, candidate_parent_mean, candidate_right_mean));
}

long double RegressionTree::computeGammaSplitValue(size_t n_left, long double sum_left, size_t n_right, long double sum_right) {
  (void) sum_left;
  (void) sum_right;
  const long double mean_parent = candidate_parent_mean;

  auto deviance = [mean_parent](long double mean, long double difference) {
    long double ratio_difference = difference / mean_parent;
    if (!isfinite(ratio_difference) ||
        ratio_difference <= -0.5L || ratio_difference > 1) {
      return 2 * (-logl(mean) + logl(mean_parent) + ratio_difference);
    }
    return -2 * log1pMinusArgument(ratio_difference);
  };

  return safeRegressionSplitScore(n_left * deviance(candidate_left_mean, candidate_left_difference) + n_right * deviance(candidate_right_mean, candidate_right_difference));
}

long double RegressionTree::computeInverseGaussianSplitValue(size_t n_left, long double sum_left, size_t n_right, long double sum_right) {
  (void) sum_left;
  (void) sum_right;
  const long double mean_parent = candidate_parent_mean;

  auto deviance = [mean_parent](
    long double mean, long double difference) {
    long double relative_difference = difference / mean_parent;
    return relative_difference * relative_difference / mean;
  };

  return safeRegressionSplitScore(
    n_left * deviance(candidate_left_mean, candidate_left_difference) +
    n_right * deviance(candidate_right_mean, candidate_right_difference));
}

long double RegressionTree::computeTweedieSplitValue(
    size_t n_left, long double sum_left, size_t n_right,
    long double sum_right) {
  const long double power = splitrule_par;
  if (power == 0) {
    return computeMSESplitValue(n_left, sum_left, n_right, sum_right);
  }
  if (power == 1) {
    return computePoissonSplitValue(n_left, sum_left, n_right, sum_right);
  }
  if (power == 2) {
    return computeGammaSplitValue(n_left, sum_left, n_right, sum_right);
  }
  if (power == 3) {
    return computeInverseGaussianSplitValue(n_left, sum_left, n_right, sum_right);
  }
  long double mean_parent = candidate_parent_mean;
  long double mean_left = candidate_left_mean;
  long double mean_right = candidate_right_mean;
  const long double q = 2 - power;

  bool constrained_boundary = false;
  if (power < 0) {
    // For xi < 0, y may be real but the mean parameter is non-negative.
    // The constrained node optimum is therefore max(sample mean, 0).
    constrained_boundary = mean_left <= 0 || mean_right <= 0 || mean_parent <= 0;
    mean_left = max(mean_left, static_cast<long double>(0));
    mean_right = max(mean_right, static_cast<long double>(0));
    mean_parent = max(mean_parent, static_cast<long double>(0));
  }

  if (constrained_boundary) {
    /*
      A daughter on the boundary is not an ordinary Bregman projection:
      clamping breaks the weighted-mean identity used by the deviance
      decomposition below. Compare the minimized node potentials directly.
      When the parent optimum is interior, use it as the common scale so this
      score is directly comparable with the normalized interior score below.
      A zero-boundary parent makes every candidate constrained, so the fixed
      node-level response scale is used instead.
    */
    long double normalizer = mean_parent > 0 ? mean_parent : tweedie_node_scale;
    long double scaled_left = mean_left / normalizer;
    long double scaled_right = mean_right / normalizer;
    long double scaled_parent = mean_parent / normalizer;
    long double potential_gain = n_left * powl(scaled_left, q) + n_right * powl(scaled_right, q) - (n_left + n_right) * powl(scaled_parent, q);
    long double log_denominator = logl(abs(q)) + logl(abs(q - 1));
    if (isfinite(potential_gain) && potential_gain > 0) {
      return logl(potential_gain) - log_denominator;
    }

    const long double negative_infinity = -numeric_limits<long double>::infinity();
    long double log_left = scaled_left > 0 ? logl(static_cast<long double>(n_left)) + q * logl(scaled_left) : negative_infinity;
    long double log_right = scaled_right > 0 ? logl(static_cast<long double>(n_right)) + q * logl(scaled_right) : negative_infinity;
    long double log_parent = scaled_parent > 0 ? logl(static_cast<long double>(n_left + n_right)) + q * logl(scaled_parent) : negative_infinity;
    return signedLogSumMagnitude(1, log_left, 1, log_right, -1, log_parent) - log_denominator;
  }

  if (mean_parent == 0) {
    return -numeric_limits<long double>::infinity();
  }

  auto deviance = [mean_parent, q](
      long double mean, long double ratio_difference) {
    if (ratio_difference == 0) {
      return -numeric_limits<long double>::infinity();
    }
    if (mean == 0) {
      return -logl(q);
    }

    long double ratio = 1 + ratio_difference;
    bool far_from_one = !isfinite(ratio) || ratio < 0.5L || ratio > 2;
    long double log_ratio = far_from_one ? logl(mean) - logl(mean_parent) : log1pl(ratio_difference);
    long double normalized_deviance;
    if (q > 0.5L) {
      // Factor q - 1 analytically. This is the stable Poisson-side form.
      long double relative_exponent = (q - 1) * log_ratio;
      if (far_from_one) {
        normalized_deviance = (ratio * log_ratio * expm1Ratio(relative_exponent) - ratio_difference) / q;
      } else {
        normalized_deviance = (log1pMinusArgument(ratio_difference) + ratio_difference * log_ratio + ratio * log_ratio * expm1RatioMinusOne(relative_exponent)) / q;
      }
    } else {
      // Factor q analytically. This is the stable Gamma-side form.
      long double relative_exponent = q * log_ratio;
      if (far_from_one) {
        normalized_deviance = (log_ratio * expm1Ratio(relative_exponent) - ratio_difference) / (q - 1);
      } else {
        normalized_deviance = (log1pMinusArgument(ratio_difference) + log_ratio * expm1RatioMinusOne(relative_exponent)) / (q - 1);
      }
    }
    if (isfinite(normalized_deviance) && normalized_deviance > 0) {
      return logl(normalized_deviance);
    }

    // B = r^q - q*r + q - 1. Evaluate this signed sum in the log
    // domain if its exponent exceeds the long-double range.
    long double log_power = q * log_ratio;
    long double log_linear = logl(abs(q)) + log_ratio;
    long double log_constant = logl(abs(q - 1));
    int linear_sign = q > 0 ? -1 : 1;
    int constant_sign = q > 1 ? 1 : -1;
    long double log_bracket = signedLogSumMagnitude(1, log_power, linear_sign, log_linear, constant_sign, log_constant);
    return log_bracket - logl(abs(q)) - logl(abs(q - 1));
  };

  long double log_left = logl(static_cast<long double>(n_left)) + deviance(mean_left, candidate_left_difference / mean_parent);
  long double log_right = logl(static_cast<long double>(n_right)) + deviance(mean_right, candidate_right_difference / mean_parent);
  return logSumExp(log_left, log_right);
}

long double RegressionTree::computeHuberSplitValue(const vector<double>& left_responses, const vector<double>& right_responses, long double parent_center) {
  long double minimum_response = min(static_cast<long double>(left_responses.front()), static_cast<long double>(right_responses.front()));
  long double maximum_response = max(static_cast<long double>(left_responses.back()), static_cast<long double>(right_responses.back()));
  if (splitrule_par >= maximum_response - minimum_response) {
    long double left_sum = 0;
    long double left_correction = 0;
    for (double response : left_responses) {
      compensatedAdd(response, left_sum, left_correction);
    }
    long double right_sum = 0;
    long double right_correction = 0;
    for (double response : right_responses) {
      compensatedAdd(response, right_sum, right_correction);
    }
    long double mean_difference = differenceOfCompensatedMeans(left_sum, left_correction, left_responses.size(), right_sum, right_correction, right_responses.size());
    long double harmonic_count = static_cast<long double>(left_responses.size()) * right_responses.size() / (left_responses.size() + right_responses.size());
    return safeRegressionSplitScore(harmonic_count * mean_difference * mean_difference);
  }

  long double left_center = computeHuberCenterSorted(left_responses);
  long double right_center = computeHuberCenterSorted(right_responses);
  long double score = 0;
  long double correction = 0;
  for (double response : left_responses) {
    compensatedAdd(huberLossDifference(response, parent_center, left_center, splitrule_par), score, correction);
  }
  for (double response : right_responses) {
    compensatedAdd(huberLossDifference(response, parent_center, right_center, splitrule_par),
      score, correction);
  }
  score += correction;
  return safeRegressionSplitScore(score);
}

bool RegressionTree::useTweediePowerSumComparison(
    long double split_value) const {
  if (splitrule_id != RegressionSplitRule::Tweedie) {
    return false;
  }
  if (!isfinite(split_value)) {
    return false;
  }
  /*
    The scalar log score is preferable while its local spacing is comfortably
    smaller than the least possible O(1 / n) count-coefficient change. Switch
    to the structured power-sum comparison only when that information is no
    longer representable. This depends on the realised score scale, not merely
    on xi: a huge power applied to nearly equal means can still have a small,
    accurately factored log score.
  */
  long double count = candidate_tweedie_parent.count;
  return abs(split_value) * LDBL_EPSILON * count >= 0.01L;
}

int RegressionTree::compareTweediePowerSums(const MeanComponents& first_left, const MeanComponents& first_right,
                                            const MeanComponents& second_left, const MeanComponents& second_right) const {
  struct SignedPowerTerm {
    MeanComponents mean;
    long double coefficient;
  };

  auto meanValue = [](const MeanComponents& value) {
    if (value.count == 0) {
      return static_cast<long double>(0);
    }
    return meanFromCompensatedSum(
      value.sum, value.correction, value.count);
  };
  auto meanDifference = [](const MeanComponents& first,
                           const MeanComponents& second) {
    return differenceOfCompensatedMeans(
      first.sum, first.correction, first.count,
      second.sum, second.correction, second.count);
  };
  auto compareMeans = [&meanDifference](
      const MeanComponents& first, const MeanComponents& second) {
    long double difference = meanDifference(first, second);
    return difference > 0 ? 1 : (difference < 0 ? -1 : 0);
  };

  SignedPowerTerm input[4] = {
    {first_left, static_cast<long double>(first_left.count)},
    {first_right, static_cast<long double>(first_right.count)},
    {second_left, -static_cast<long double>(second_left.count)},
    {second_right, -static_cast<long double>(second_right.count)}
  };
  SignedPowerTerm grouped[4];
  size_t num_grouped = 0;
  for (const auto& term : input) {
    if (term.mean.count == 0 || term.coefficient == 0) {
      continue;
    }
    size_t group = 0;
    while (group < num_grouped &&
           compareMeans(term.mean, grouped[group].mean) != 0) {
      ++group;
    }
    if (group == num_grouped) {
      grouped[num_grouped++] = term;
    } else {
      grouped[group].coefficient += term.coefficient;
    }
  }

  size_t output = 0;
  for (size_t i = 0; i < num_grouped; ++i) {
    if (grouped[i].coefficient != 0) {
      grouped[output++] = grouped[i];
    }
  }
  num_grouped = output;
  if (num_grouped == 0) {
    return 0;
  }

  const long double q = 2 - static_cast<long double>(splitrule_par);
  // Put the power-dominant feasible mean first: largest for q > 0 and
  // smallest for q < 0.
  for (size_t i = 1; i < num_grouped; ++i) {
    SignedPowerTerm current = grouped[i];
    size_t position = i;
    while (position > 0) {
      int comparison = compareMeans(current.mean, grouped[position - 1].mean);
      bool dominates = q > 0 ? comparison > 0 : comparison < 0;
      if (!dominates) {
        break;
      }
      grouped[position] = grouped[position - 1];
      --position;
    }
    grouped[position] = current;
  }

  const MeanComponents& anchor = grouped[0].mean;
  long double anchor_mean = meanValue(anchor);
  if (anchor_mean == 0) {
    return 0;
  }

  long double normalized_difference = 0;
  long double correction = 0;
  for (size_t i = 0; i < num_grouped; ++i) {
    long double current_mean = meanValue(grouped[i].mean);
    long double exponent;
    if (current_mean == 0) {
      exponent = -numeric_limits<long double>::infinity();
    } else {
      long double difference = meanDifference(grouped[i].mean, anchor);
      long double relative_difference = difference / anchor_mean;
      long double log_ratio =
        relative_difference > -0.5L && relative_difference < 1 ?
        log1pl(relative_difference) :
        logl(current_mean) - logl(anchor_mean);
      exponent = q * log_ratio;
      if (exponent > 0) {
        exponent = 0;
      }
    }
    compensatedAdd(grouped[i].coefficient * expm1l(exponent), normalized_difference, correction);
  }
  normalized_difference += correction;
  return normalized_difference > 0 ? 1 : (normalized_difference < 0 ? -1 : 0);
}

bool RegressionTree::isBetterSplit(
    long double split_value, long double best_split_value, bool found_split) {
  bool candidate_uses_power_sum =
    useTweediePowerSumComparison(split_value);
  if (!candidate_uses_power_sum && !isfinite(split_value)) {
    return false;
  }

  if (candidate_uses_power_sum) {
    MeanComponents empty;
    int gain_sign = compareTweediePowerSums(
      candidate_tweedie_left, candidate_tweedie_right,
      candidate_tweedie_parent, empty);
    // A representability-triggered comparison is far outside q in (0, 1),
    // where the power-sum denominator is positive.
    if (gain_sign <= 0) {
      return false;
    }
  }

  bool better;
  if (!found_split) {
    better = true;
  } else if (candidate_uses_power_sum || best_tweedie_uses_power_sum) {
    better = compareTweediePowerSums(
      candidate_tweedie_left, candidate_tweedie_right,
      best_tweedie_left, best_tweedie_right) > 0;
  } else {
    better = split_value > best_split_value;
  }
  if (!better) {
    return false;
  }
  best_tweedie_left = candidate_tweedie_left;
  best_tweedie_right = candidate_tweedie_right;
  best_tweedie_uses_power_sum = candidate_uses_power_sum;
  return true;
}

long double RegressionTree::computeSplitValue(size_t n_left, long double sum_left, size_t n_right, long double sum_right, long double sum_correction_left,
                                              long double sum_correction_right, long double failure_sum_left, long double failure_sum_right,
                                              long double failure_correction_left, long double failure_correction_right, const vector<double>& left_responses,
                                              const vector<double>& right_responses, long double parent_robust_center) {
  candidate_left_mean = meanFromCompensatedSum(sum_left, sum_correction_left, n_left);
  candidate_right_mean = meanFromCompensatedSum(sum_right, sum_correction_right, n_right);

  long double total_sum = 0;
  long double total_correction = 0;
  compensatedAdd(sum_left, total_sum, total_correction);
  compensatedAdd(sum_right, total_sum, total_correction);
  compensatedAdd(sum_correction_left, total_sum, total_correction);
  compensatedAdd(sum_correction_right, total_sum, total_correction);
  candidate_parent_mean = meanFromCompensatedSum(total_sum, total_correction, n_left + n_right);
  candidate_tweedie_left = {
    sum_left, sum_correction_left, n_left
  };
  candidate_tweedie_right = {
    sum_right, sum_correction_right, n_right
  };
  candidate_tweedie_parent = {
    total_sum, total_correction, n_left + n_right
  };
  if (splitrule_id == RegressionSplitRule::Tweedie && splitrule_par < 0) {
    if (candidate_left_mean <= 0) {
      candidate_tweedie_left = {0, 0, n_left};
    }
    if (candidate_right_mean <= 0) {
      candidate_tweedie_right = {0, 0, n_right};
    }
    if (candidate_parent_mean <= 0) {
      candidate_tweedie_parent = {0, 0, n_left + n_right};
    }
  }

  long double mean_difference = differenceOfCompensatedMeans(sum_left, sum_correction_left, n_left, sum_right, sum_correction_right, n_right);
  candidate_left_difference = static_cast<long double>(n_right) / (n_left + n_right) * mean_difference;
  candidate_right_difference = -static_cast<long double>(n_left) / (n_left + n_right) * mean_difference;

  candidate_left_failure_mean = meanFromCompensatedSum(failure_sum_left, failure_correction_left, n_left);
  candidate_right_failure_mean = meanFromCompensatedSum(failure_sum_right, failure_correction_right, n_right);
  total_sum = 0;
  total_correction = 0;
  compensatedAdd(failure_sum_left, total_sum, total_correction);
  compensatedAdd(failure_sum_right, total_sum, total_correction);
  compensatedAdd(failure_correction_left, total_sum, total_correction);
  compensatedAdd(failure_correction_right, total_sum, total_correction);
  candidate_failure_parent_mean = meanFromCompensatedSum(total_sum, total_correction, n_left + n_right);
  long double failure_mean_difference = differenceOfCompensatedMeans(failure_sum_left, failure_correction_left, n_left, failure_sum_right, failure_correction_right, n_right);
  candidate_left_failure_difference = static_cast<long double>(n_right) / (n_left + n_right) * failure_mean_difference;
  candidate_right_failure_difference = -static_cast<long double>(n_left) / (n_left + n_right) * failure_mean_difference;

  switch (splitrule_id) {
    case RegressionSplitRule::MSE:
      return computeMSESplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::MAE:
      return computeMAESplitValue(left_responses, right_responses, parent_robust_center);
    case RegressionSplitRule::Binomial:
      return computeBinomialSplitValue(n_left, sum_left, failure_sum_left, n_right, sum_right, failure_sum_right);
    case RegressionSplitRule::NegativeBinomial:
      return computeNegativeBinomialSplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::Poisson:
      return computePoissonSplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::Gamma:
      return computeGammaSplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::InverseGaussian:
      return computeInverseGaussianSplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::Tweedie:
      return computeTweedieSplitValue(n_left, sum_left, n_right, sum_right);
    case RegressionSplitRule::Huber:
      return computeHuberSplitValue(left_responses, right_responses, parent_robust_center);
  }
  return -numeric_limits<long double>::infinity();
}

void RegressionTree::reserveTreeMemory(size_t num_obs) {
  // the minimal node size gives an upper bound for the number of nodes in the tree
  size_t max_terminal_nodes = min_node_size == 0 ? max(static_cast<size_t>(1), num_obs) :
    max(static_cast<size_t>(1), num_obs / min_node_size);
  size_t max_num_nodes = 2 * max_terminal_nodes - 1;
  // avoid excessive reservation for unusually large shallow trees
  max_num_nodes = min(max_num_nodes, static_cast<size_t>(1024));

  node_obs.reserve(max_num_nodes);
  if (honest) {
    holdout_node_obs.reserve(max_num_nodes);
  }
  node_sizes.reserve(max_num_nodes);
  left_daughters.reserve(max_num_nodes);
  feature_IDs.reserve(max_num_nodes);
  thresholds.reserve(max_num_nodes);
  depths.reserve(max_num_nodes);
  means.reserve(max_num_nodes);
  sum_node.reserve(max_num_nodes);
}

void RegressionTree::bestSplitContinuous(
    size_t node_index, size_t feature,
    const vector<size_t>& response_order, long double parent_robust_center,
    long double& best_split_val, bool& found_split, size_t& best_feature,
    vector<double>& best_threshold, long double& best_sum_left) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  size_t num_obs_parent = current_node_obs.size();

  // samples split points
  vector<double> split_points;
  size_t nsplits_final = sampleSplitPoints(split_points, current_node_obs, feature);

  // no possible splits
  if (nsplits_final == 0) {
    return;
  }

  /*
    Bucket each observation by the first threshold whose left daughter
    contains it. Prefix/suffix sweeps then recover independent compensated
    sums for both daughters in O(n log(nsplits) + nsplits), without a
    cancellation-prone parent-minus-daughter subtraction.
  */
  num_obs_right.assign(nsplits_final, 0);
  sums_right.assign(nsplits_final, 0);
  vector<long double> sums_left(nsplits_final, 0);
  vector<long double> sum_corrections_left(nsplits_final, 0);
  vector<long double> sum_corrections_right(nsplits_final, 0);
  vector<size_t> bucket_counts(nsplits_final + 1, 0);
  vector<long double> bucket_sums(nsplits_final + 1, 0);
  vector<long double> bucket_corrections(nsplits_final + 1, 0);
  bool use_failure_sums = splitrule_id == RegressionSplitRule::Binomial;
  vector<long double> failure_sums_left(use_failure_sums ? nsplits_final : 0, 0);
  vector<long double> failure_sums_right(use_failure_sums ? nsplits_final : 0, 0);
  vector<long double> failure_corrections_left(use_failure_sums ? nsplits_final : 0, 0);
  vector<long double> failure_corrections_right(use_failure_sums ? nsplits_final : 0, 0);
  vector<long double> bucket_failure_sums(use_failure_sums ? nsplits_final + 1 : 0, 0);
  vector<long double> bucket_failure_corrections(use_failure_sums ? nsplits_final + 1 : 0, 0);
  for (size_t i : current_node_obs) {
    double feature_val = data->get_x(i, feature);
    double response_val = data->get_y(i);
    size_t bucket = std::isnan(feature_val) ? nsplits_final : lower_bound(split_points.begin(), split_points.end(), feature_val) - split_points.begin();
    ++bucket_counts[bucket];
    compensatedAdd(response_val, bucket_sums[bucket], bucket_corrections[bucket]);
    if (use_failure_sums) {
      compensatedAdd(1 - static_cast<long double>(response_val), bucket_failure_sums[bucket], bucket_failure_corrections[bucket]);
    }
  }

  long double current_sum = 0;
  long double current_correction = 0;
  long double current_failure_sum = 0;
  long double current_failure_correction = 0;
  for (size_t s = 0; s < nsplits_final; ++s) {
    compensatedAdd(bucket_sums[s], current_sum, current_correction);
    compensatedAdd(bucket_corrections[s], current_sum, current_correction);
    sums_left[s] = current_sum;
    sum_corrections_left[s] = current_correction;
    if (use_failure_sums) {
      compensatedAdd(bucket_failure_sums[s], current_failure_sum, current_failure_correction);
      compensatedAdd(bucket_failure_corrections[s], current_failure_sum, current_failure_correction);
      failure_sums_left[s] = current_failure_sum;
      failure_corrections_left[s] = current_failure_correction;
    }
  }

  size_t current_count_right = 0;
  current_sum = 0;
  current_correction = 0;
  current_failure_sum = 0;
  current_failure_correction = 0;
  for (size_t bucket = nsplits_final; bucket > 0; --bucket) {
    current_count_right += bucket_counts[bucket];
    compensatedAdd(bucket_sums[bucket], current_sum, current_correction);
    compensatedAdd(bucket_corrections[bucket], current_sum, current_correction);
    size_t s = bucket - 1;
    num_obs_right[s] = current_count_right;
    sums_right[s] = current_sum;
    sum_corrections_right[s] = current_correction;
    if (use_failure_sums) {
      compensatedAdd(bucket_failure_sums[bucket], current_failure_sum, current_failure_correction);
      compensatedAdd(bucket_failure_corrections[bucket], current_failure_sum, current_failure_correction);
      failure_sums_right[s] = current_failure_sum;
      failure_corrections_right[s] = current_failure_correction;
    }
  }

  bool use_robust_loss = splitrule_id == RegressionSplitRule::MAE || splitrule_id == RegressionSplitRule::Huber;
  vector<double> left_responses;
  vector<double> right_responses;
  if (use_robust_loss) {
    left_responses.reserve(num_obs_parent);
    right_responses.reserve(num_obs_parent);
  }

  // now find the best split
  for (size_t s = 0; s < nsplits_final; ++s) {
    // if one of the daughter nodes are too small, skip the computation for that split
    size_t num_obs_left = num_obs_parent - num_obs_right[s];
    if (num_obs_right[s] < min_node_size || num_obs_left < min_node_size) {
      continue;
    }

    long double sum_left = sums_left[s];
    if (use_robust_loss) {
      left_responses.clear();
      right_responses.clear();
      // response_order is sorted by y, so both stable partitions remain sorted
      for (size_t i : response_order) {
        if (data->get_x(i, feature) <= split_points[s]) {
          left_responses.push_back(data->get_y(i));
        } else {
          right_responses.push_back(data->get_y(i));
        }
      }
    }
    long double split_val = computeSplitValue(num_obs_left, sum_left, num_obs_right[s], sums_right[s], sum_corrections_left[s], sum_corrections_right[s],
                                              use_failure_sums ? failure_sums_left[s] : 0, use_failure_sums ? failure_sums_right[s] : 0,
                                              use_failure_sums ? failure_corrections_left[s] : 0, use_failure_sums ? failure_corrections_right[s] : 0,
                                              left_responses, right_responses, parent_robust_center);

    if (isBetterSplit(split_val, best_split_val, found_split)) {
      best_split_val = split_val;
      found_split = true;
      best_feature = feature;
      best_sum_left = sum_left + sum_corrections_left[s];
      best_threshold = {split_points[s]};
    }
  }
}

void RegressionTree::bestSplitCategorical(size_t node_index, size_t feature, long double& best_split_val, bool& found_split, 
                                          size_t& best_feature, vector<double>& best_threshold, long double& best_sum_left,
                                          const vector<size_t>& response_order, long double parent_robust_center) {
  const vector<size_t>& current_node_obs = node_obs[node_index];
  vector<double> feature_values = data->getValues(current_node_obs, feature);
  bool has_missing_values = any_of(feature_values.begin(), feature_values.end(), [](double value) { return std::isnan(value); });
  feature_values.erase(remove_if(feature_values.begin(), feature_values.end(), [](double value) { return std::isnan(value); }), feature_values.end());
  feature_values = uniqueValues(std::move(feature_values));
  size_t num_feature_values = feature_values.size();

  unordered_set<uint64_t> partition_masks;
  // generate partitions (breaks if no possible splits)
  if (generateCategoricalPartitions(feature_values, partition_masks, has_missing_values)) {
      return;
  }

  // compute the number of observations and sum of responses for each category once
  vector<size_t> category_counts(num_feature_values, 0);
  vector<long double> category_sums(num_feature_values, 0);
  vector<long double> category_corrections(num_feature_values, 0);
  bool use_failure_sums = splitrule_id == RegressionSplitRule::Binomial;
  vector<long double> category_failure_sums(use_failure_sums ? num_feature_values : 0, 0);
  vector<long double> category_failure_corrections(use_failure_sums ? num_feature_values : 0, 0);
  long double missing_sum = 0;
  long double missing_correction = 0;
  long double missing_failure_sum = 0;
  long double missing_failure_correction = 0;
  for (size_t obs_id : current_node_obs) {
    double feature_value = data->get_x(obs_id, feature);
    // missing feature values are sent to the right daughter during prediction
    size_t category = std::isnan(feature_value) ? num_feature_values : lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin();
    if (category < num_feature_values) {
      ++category_counts[category];
      compensatedAdd(data->get_y(obs_id), category_sums[category], category_corrections[category]);
      if (use_failure_sums) {
        compensatedAdd(1 - static_cast<long double>(data->get_y(obs_id)), category_failure_sums[category], category_failure_corrections[category]);
      }
    } else {
      compensatedAdd(data->get_y(obs_id), missing_sum, missing_correction);
      if (use_failure_sums) {
        compensatedAdd(1 - static_cast<long double>(data->get_y(obs_id)), missing_failure_sum, missing_failure_correction);
      }
    }
  }

  bool use_robust_loss = splitrule_id == RegressionSplitRule::MAE || splitrule_id == RegressionSplitRule::Huber;
  vector<double> left_responses;
  vector<double> right_responses;
  if (use_robust_loss) {
    left_responses.reserve(current_node_obs.size());
    right_responses.reserve(current_node_obs.size());
  }

  // consider each partition (bitmask)
  for (const auto& mask : partition_masks) {
    size_t n_left = 0;
    long double sum_left = 0;
    long double correction_left = 0;
    long double sum_right = missing_sum;
    long double correction_right = missing_correction;
    long double failure_sum_left = 0;
    long double failure_correction_left = 0;
    long double failure_sum_right = missing_failure_sum;
    long double failure_correction_right = missing_failure_correction;
    for (size_t i = 0; i < num_feature_values; ++i) {
        if (i < 63 && ((mask >> i) & 1)) {
            n_left += category_counts[i];
            compensatedAdd(category_sums[i], sum_left, correction_left);
            compensatedAdd(category_corrections[i], sum_left, correction_left);
            if (use_failure_sums) {
              compensatedAdd(category_failure_sums[i], failure_sum_left, failure_correction_left);
              compensatedAdd(category_failure_corrections[i], failure_sum_left, failure_correction_left);
            }
        } else {
            compensatedAdd(category_sums[i], sum_right, correction_right);
            compensatedAdd(category_corrections[i], sum_right, correction_right);
            if (use_failure_sums) {
              compensatedAdd(category_failure_sums[i], failure_sum_right, failure_correction_right);
              compensatedAdd(category_failure_corrections[i], failure_sum_right, failure_correction_right);
            }
        }
    }
    size_t n_right = current_node_obs.size() - n_left;

    if (n_left < min_node_size || n_right < min_node_size) {
        continue;
    }

    if (use_robust_loss) {
      left_responses.clear();
      right_responses.clear();
      for (size_t obs_id : response_order) {
        double feature_value = data->get_x(obs_id, feature);
        size_t category = std::isnan(feature_value) ? num_feature_values : lower_bound(feature_values.begin(), feature_values.end(), feature_value) - feature_values.begin();
        if (category < num_feature_values && category < 63 && ((mask >> category) & 1)) {
          left_responses.push_back(data->get_y(obs_id));
        } else {
          right_responses.push_back(data->get_y(obs_id));
        }
      }
    }
    long double split_val = computeSplitValue(n_left, sum_left, n_right, sum_right, correction_left, correction_right, failure_sum_left, failure_sum_right,
                                              failure_correction_left, failure_correction_right, left_responses, right_responses, parent_robust_center);

    if (isBetterSplit(split_val, best_split_val, found_split)) {
        best_split_val = split_val;
        found_split = true;
        best_feature = feature;
        best_threshold.clear();
        for (size_t i = 0; i < num_feature_values; ++i) {
          if (i < 63 && ((mask >> i) & 1)) {
            best_threshold.push_back(feature_values[i]);
          }
        }
        best_sum_left = sum_left + correction_left;
    }
  }
}

void RegressionTree::makeLeaf(size_t node_index) {
  // update tree info
  feature_IDs.push_back(0);
  thresholds.push_back({});

  // since we are in a terminal node, we save the indices for the observations
  for (size_t i : node_obs[node_index]) {
      prediction_node_IDs[i] = node_index;
  }
  if (honest) {
    for (size_t i : holdout_node_obs[node_index]) {
      prediction_node_IDs[i] = node_index;
    }
  }

  // the observation indices are no longer needed after the node is made terminal
  vector<size_t>().swap(node_obs[node_index]);
  if (honest) {
    vector<size_t>().swap(holdout_node_obs[node_index]);
  }
}

// function to create a split for a regression tree. returns true if leaf, otherwise false
bool RegressionTree::createSplit(size_t node_index) {
    // if we are in the root node, the sum of the responses needs to be computed
    if (sum_node.empty()) {
      reserveTreeMemory(node_obs[node_index].size());
      splitrule_id = regressionSplitRuleFromString(splitrule);
      if (regressionSplitRuleUsesParameter(splitrule_id) && (!isfinite(splitrule_par) || ((splitrule_id == RegressionSplitRule::NegativeBinomial ||
                                           splitrule_id == RegressionSplitRule::Huber) && splitrule_par <= 0) || (splitrule_id == RegressionSplitRule::Tweedie &&
                                           splitrule_par > 0 && splitrule_par < 1))) {
        throw runtime_error("Invalid parameter for parameterised regression splitrule");
      }
    }
    const vector<size_t>& current_node_obs = node_obs[node_index];
    if (sum_node.empty()) {
      sum_node.push_back(computeSum(current_node_obs));

      size_t num_features = data->getNumberOfFeatures();
      feature_indices.resize(num_features);
      for (size_t i = 0; i < num_features; ++i) {
        feature_indices[i] = i;
      }
    }
    // Recompute once from this node's observations. Propagating the right sum
    // by repeated parent-minus-left subtraction can accumulate cancellation in
    // deep trees with mixed-scale responses.
    long double parent_sum = computeSum(current_node_obs);
    sum_node[node_index] = parent_sum;

    // if no split is possible, make the node a leaf
    if (current_node_obs.size() < 2 * min_node_size) {
        if (!honest) {
            means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, we may simply reuse the computed sum
        } else {
            const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
            if (holdout_obs.empty()) {
                means.push_back(parent_sum / (double) node_sizes[node_index]);
            } else {
                means.push_back(computeSum(holdout_obs) / (double) holdout_obs.size());  // for honest trees, compute the mean from scratch for the holdout indices
            }
        }
        makeLeaf(node_index);
        return true;
    }

    // sample mtry features
    vector<size_t> sampled_features = sampleIndices(feature_indices, mtry, false, random_number_generator);
    const vector<bool>& categorical = data->getCategorical();

    // Robust criteria need the actual daughter responses. Sort the parent once
    // and stable-partition that order for every candidate, avoiding repeated sorts.
    bool use_robust_loss = splitrule_id == RegressionSplitRule::MAE || splitrule_id == RegressionSplitRule::Huber;
    vector<size_t> response_order;
    long double parent_robust_center = 0;
    if (use_robust_loss) {
      response_order = current_node_obs;
      stable_sort(response_order.begin(), response_order.end(), [this](size_t first, size_t second) {
          return data->get_y(first) < data->get_y(second);
        });
      vector<double> sorted_responses;
      sorted_responses.reserve(response_order.size());
      for (size_t observation : response_order) {
        sorted_responses.push_back(data->get_y(observation));
      }
      parent_robust_center = splitrule_id == RegressionSplitRule::MAE ? computeMedianSorted(sorted_responses) : computeHuberCenterSorted(sorted_responses);
    }
    if (splitrule_id == RegressionSplitRule::Tweedie && splitrule_par < 0) {
      tweedie_node_scale = 0;
      for (size_t observation : current_node_obs) {
        tweedie_node_scale = max(tweedie_node_scale, abs(static_cast<long double>(data->get_y(observation))));
      }
      if (tweedie_node_scale == 0) {
        tweedie_node_scale = 1;
      }
    }

    long double best_split_val = -numeric_limits<long double>::infinity();
    bool found_split = false;
    size_t best_feature = 0;
    vector<double> best_threshold;
    vector<size_t> best_left_indices;
    vector<size_t> best_right_indices;
    long double best_sum_left;
    
    // only used for honesty
    vector<size_t> holdout_left_indices;
    vector<size_t> holdout_right_indices;

    // now consider each of the sampled features
    for (size_t i : sampled_features) {
        if (categorical[i]) {
            bestSplitCategorical(node_index, i, best_split_val, found_split, best_feature, best_threshold, best_sum_left, response_order, parent_robust_center);
        }
        else {
            // does not return the best indices, so this has to be done later
            bestSplitContinuous(node_index, i, response_order, parent_robust_center, best_split_val, found_split, best_feature, best_threshold, best_sum_left);
        }
    }

    // if no best split is found, make the node a leaf
    if (!found_split) {
        if (!honest) {
          means.push_back(parent_sum / (double) node_sizes[node_index]);  // for dishonest trees, use parent info already computed earlier
        } else {
          const vector<size_t>& holdout_obs = holdout_node_obs[node_index];
          if (holdout_obs.empty()) {
            means.push_back(parent_sum / (double) node_sizes[node_index]);
          } else {
            means.push_back(computeSum(holdout_obs) / (double) holdout_obs.size()); // for honest trees, use the holdout set for computing the mean
          }
        }
        makeLeaf(node_index);
        return true;
    }

    // construct the indices of the two daughters once for the best split
    bool categorical_split = categorical[best_feature];
    best_left_indices.reserve(current_node_obs.size());
    best_right_indices.reserve(current_node_obs.size());
    for (size_t i : current_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        bool goes_left = categorical_split ?
          find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() :
          feature_value <= best_threshold[0];
        if (goes_left) {
            best_left_indices.push_back(i);
        } else {
            best_right_indices.push_back(i);
        }
    }

    // update the holdout index sets if the tree is honest
    if (honest) {
      const vector<size_t>& current_holdout_node_obs = holdout_node_obs[node_index];
      holdout_left_indices.reserve(current_holdout_node_obs.size());
      holdout_right_indices.reserve(current_holdout_node_obs.size());
      for (size_t i : current_holdout_node_obs) {
        double feature_value = data->get_x(i, best_feature);
        bool goes_left = categorical_split ?
          find(best_threshold.begin(), best_threshold.end(), feature_value) != best_threshold.end() :
          feature_value <= best_threshold[0];
        if (goes_left) {
          holdout_left_indices.push_back(i);
        } else {
          holdout_right_indices.push_back(i);
        }
      }
    }
    
    // a best split was found, update the tree
    size_t best_left_size = best_left_indices.size();
    size_t best_right_size = best_right_indices.size();
    node_obs.push_back(std::move(best_left_indices));          // construct left daughter
    node_obs.push_back(std::move(best_right_indices));         // construct right daughter
    sum_node.push_back(best_sum_left);                         // save sums of responses
    sum_node.push_back(parent_sum - best_sum_left);
    node_sizes.push_back(best_left_size);
    node_sizes.push_back(best_right_size);
    feature_IDs.push_back(best_feature);
    thresholds.push_back(std::move(best_threshold));
    means.push_back(0);

    // for honest trees, update the holdout indices
    if (honest) {
        holdout_node_obs.push_back(std::move(holdout_left_indices));
        holdout_node_obs.push_back(std::move(holdout_right_indices));
    }

    // the parent indices are no longer needed after its daughters have been constructed
    vector<size_t>().swap(node_obs[node_index]);
    if (honest) {
      vector<size_t>().swap(holdout_node_obs[node_index]);
    }

    return false;
}

// splitting rules for regression trees
//--------------------------------------------------------------------------------------

// functions related to splitting rules should be moved to here eventually, possibly refactored and with new names

// prediction for regression trees
//--------------------------------------------------------------------------------------

vector<double> RegressionTree::computePredictions(const Data& new_data) {
  size_t num_obs = new_data.getNumberOfObs();
  vector<double> predictions(num_obs);
  for (size_t i = 0; i < num_obs; ++i) {
    predictions[i] = predictValue(new_data, i);
  }
  return predictions;
}

// error estimation for regression trees
//--------------------------------------------------------------------------------------

// computes the mean squared error based on a vector of predictions and a test vector response
double computeMSE(const vector<double>& predictions, const vector<double>& response) {
  size_t n = predictions.size();
  double ssq = 0;
  for (size_t i = 0; i < n; ++i) {
    ssq += (predictions[i] - response[i]) * (predictions[i] - response[i]);
  }
  return ssq / (double) n;
}

// computes the R^2 error based on MSE
double computeR2(double mse, const vector<double>& response) {
  size_t n = response.size();

  // compute the mse for the pure intercept model
  double response_mean = vector_sum(response) / (double) n;
  double null_ssq = 0;
  for (size_t i = 0; i < n; ++i) {
    null_ssq += (response_mean - response[i]) * (response_mean - response[i]);
  }
  double null_mse = null_ssq / (double) n;
  return 1 - mse / null_mse;
}
