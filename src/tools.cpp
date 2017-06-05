#include <iostream>
#include <math.h>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
    * Calculate the RMSE here.
    */

  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // check the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  if (estimations.size() != ground_truth.size() || estimations.size() == 0) {
    cout << "Invalid estimation or ground_truth data" << endl;
    return rmse;
  }

  //accumulate squared residuals
  for (unsigned int i = 0; i < estimations.size(); ++i) {
    VectorXd residual = estimations[i] - ground_truth[i];

    // Coefficient-wise multiplication
    residual = residual.array() * residual.array();
    rmse += residual;
  }

  // Calculate the mean
  rmse = rmse / estimations.size();

  // Calculate the squared root
  rmse = rmse.array().sqrt();

  // Return the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd &x_state) {
  /**
   * Calculate a Jacobian here.
   */

  MatrixXd Hj(3, 4);

  // Recover state parameters
  double px = x_state(0);
  double py = x_state(1);

  // Check division by zero
  if (px == 0 && py == 0) {
    cout << "CalculateJacobian () - Error - Division by Zero" << endl;
    return Hj;
  }

  // Recover vx, vy after check px, py
  double vx = x_state(2);
  double vy = x_state(3);

  // Compute the Jacobian matrix
  double square_sum = pow(px, 2) + pow(py, 2);
  double sss = sqrt(square_sum);  // sqrt_square_sum
  double pss = square_sum * sss;  // pow_square_sum_1p5

  Hj << px / sss, py / sss, 0, 0,
      -py / square_sum, px / square_sum, 0, 0,
      py * (vx * py - vy * px) / pss, px * (vy * px - vx * py) / pss, px / sss, py / sss;

  return Hj;
}
