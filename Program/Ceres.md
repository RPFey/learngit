# Ceres

## minimization

build a functor to evaluate the specific function f(x)

```c++
struct CostFunctor {
   template <typename T>
   bool operator()(const T* const x, T* residual) const {
     residual[0] = T(10.0) - x[0]; // x 是需要优化的参数组成的数组
     return true;
   }
};

// Build the problem.
Problem problem;

// Set up the only cost function (also known as residual). This uses
// auto-differentiation to obtain the derivative (jacobian).
CostFunction* cost_function =
    new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
    // 两个参数是输出维度(residual)与输入维度 (x)
problem.AddResidualBlock(cost_function, NULL, &x);
```

当损失函数中需要调用其它库中的函数，可以使用数值求导(numerical derivatives)

```c++
struct NumericDiffCostFunctor {
  bool operator()(const double* const x, double* residual) const {
    residual[0] = 10.0 - x[0]; // 指定了类型
    return true;
  }
};
CostFunction* cost_function =
  new NumericDiffCostFunction<NumericDiffCostFunctor, ceres::CENTRAL, 1, 1>(
      // 多加了一个参数
      new NumericDiffCostFunctor);
problem.AddResidualBlock(cost_function, NULL, &x);
```

min  || F(x) || 中F(x)  是一个向量值函数 F(x) = [f1, f2, f3, f4]

 只需要分别构建这四个函数的 CostFunctor 之后再依次添加进 AddResidualBlock 中即可

```c++
// Add residual terms to the problem using the using the autodiff
// wrapper to get the derivatives automatically.
problem.AddResidualBlock(
  new AutoDiffCostFunction<F1, 1, 1, 1>(new F1), NULL, &x1, &x2);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F2, 1, 1, 1>(new F2), NULL, &x3, &x4);
problem.AddResidualBlock(
  new AutoDiffCostFunction<F3, 1, 1, 1>(new F3), NULL, &x2, &x3)
problem.AddResidualBlock(
  new AutoDiffCostFunction<F4, 1, 1, 1>(new F4), NULL, &x1, &x4);
// F1 - F4 为 functor
```

## curve fitting

做曲线拟合时，就是把每一个数据点整合在一起做一次最小化

```c++
struct ExponentialResidual {
  ExponentialResidual(double x, double y)
      : x_(x), y_(y) {}

  template <typename T>
  bool operator()(const T* const m, const T* const c, T* residual) const {
    residual[0] = T(y_) - exp(m[0] * T(x_) + c[0]);
    return true;
  }

 private:
  // Observations for a sample. x_, y_ are data points
  const double x_;
  const double y_;
};

double m = 0.0;
double c = 0.0;

Problem problem;
for (int i = 0; i < kNumObservations; ++i) {
  // 在每一个数据点处加入一个residual block
  CostFunction* cost_function =
       new AutoDiffCostFunction<ExponentialResidual, 1, 1, 1>(
           new ExponentialResidual(data[2 * i], data[2 * i + 1]));
  problem.AddResidualBlock(cost_function, NULL, &m, &c);
}
```

Loss Function (remove outliers)

```c++
problem.AddResidualBlock(cost_function, new CauchyLoss(0.5) , &m, &c);
```

add Cauchy loss as the kernel function to remove outliers.

相机参数估计：

\1. 从世界坐标变换到相机坐标(相机观测到的数据，如果是相机系，则省略这一步)

\2. 畸变矫正

\3. 变换回世界坐标与 ground truth 计算误差

```c++
struct SnavelyReprojectionError {
  SnavelyReprojectionError(double observed_x, double observed_y)
      : observed_x(observed_x), observed_y(observed_y) {}

  template <typename T>
  bool operator()(const T* const camera,
                  const T* const point,
                  T* residuals) const {
    // camera[0,1,2] are the angle-axis rotation.
    T p[3];
    ceres::AngleAxisRotatePoint(camera, point, p);
    // camera[3,4,5] are the translation.
    p[0] += camera[3]; p[1] += camera[4]; p[2] += camera[5];

    // Compute the center of distortion. The sign change comes from
    // the camera model that Noah Snavely's Bundler assumes, whereby
    // the camera coordinate system has a negative z axis.
    T xp = - p[0] / p[2];
    T yp = - p[1] / p[2];

    // Apply second and fourth order radial distortion.
    const T& l1 = camera[7];
    const T& l2 = camera[8];
    T r2 = xp*xp + yp*yp;
    T distortion = T(1.0) + r2  * (l1 + l2  * r2);

    // Compute final projected point position.
    const T& focal = camera[6];
    T predicted_x = focal * distortion * xp;
    T predicted_y = focal * distortion * yp;

    // The error is the difference between the predicted and observed position.
    residuals[0] = predicted_x - T(observed_x);
    residuals[1] = predicted_y - T(observed_y);
    return true;
  }

   // Factory to hide the construction of the CostFunction object from
   // the client code.
   static ceres::CostFunction* Create(const double observed_x,
                                      const double observed_y) {
     return (new ceres::AutoDiffCostFunction<SnavelyReprojectionError, 2, 9, 3>(
         // because the imput parameters camera/point are 9/3 dimensions
                 new SnavelyReprojectionError(observed_x, observed_y)));
   }
  
  // ground truth data
  double observed_x;
  double observed_y;
};
```

optimization options :

```c++
options.linear_solver_type = ceres::DENSE_SCHUR;
// alternatives : ceres::SPARSE_NORMAL_CHOLESKY for sparse matrix
```
