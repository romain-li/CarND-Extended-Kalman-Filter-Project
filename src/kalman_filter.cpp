#include "kalman_filter.h"
#include <math.h>

using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::Vector3d;

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}

void KalmanFilter::Predict() {
  /**
    * predict the state
    */

  x_ = F_ * x_;
  P_ = F_ * P_ * F_.transpose() + Q_;
}

void KalmanFilter::BaseUpdate(const VectorXd &y) {
  MatrixXd Ht = H_.transpose();
  MatrixXd S = H_ * P_ * Ht + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = P_ * Ht * Si;

  // New estimate
  x_ = x_ + (K * y);
  long x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
}

void KalmanFilter::Update(const VectorXd &z) {
  /**
    * update the state by using Kalman Filter equations
    */

  VectorXd y = z - H_ * x_;

  BaseUpdate(y);
}

void KalmanFilter::UpdateEKF(const VectorXd &z) {
  /**
    * update the state by using Extended Kalman Filter equations
    */

  double px = x_(0);
  double py = x_(1);
  double vx = x_(2);
  double vy = x_(3);
  VectorXd hx = VectorXd(3);
  double square_sum = pow(px, 2) + pow(py, 2);
  double sss = sqrt(square_sum);  // sqrt_square_sum
  hx << sss, atan2(py, px), (px * vx + py * vy) / sss;

  VectorXd y = z - hx;

  // Normalize y matrix to make sure we can get the correct x after update function.
  if (y(1) > M_PI) {
    y(1) -= 2 * M_PI;
  } else if (y(1) < -M_PI) {
    y(1) += 2 * M_PI;
  }

  BaseUpdate(y);
}
