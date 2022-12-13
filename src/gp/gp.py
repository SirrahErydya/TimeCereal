import gpytorch


class PPolynomialGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(PPolynomialGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.PiecewisePolynomialKernel(3)  # Construct the kernel function

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


class MaternGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(MaternGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.ScaleKernel(gpytorch.kernels.MaternKernel(nu=0.5))  # Construct the kernel function

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


class PeriodicGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(PeriodicGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.PeriodicKernel()  # Construct the kernel function

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x)


class RBFGP(gpytorch.models.ExactGP):
    def __init__(self, x_train, y_train, likelihood):
        super(RBFGP, self).__init__(x_train, y_train, likelihood)
        self.mean = gpytorch.means.ConstantMean()  # Construct the mean function
        self.cov = gpytorch.kernels.RBFKernel()  # Construct the kernel function

    def forward(self, x):
        # Evaluate the mean and kernel function at x
        mean_x = self.mean(x)
        cov_x = self.cov(x)
        # Return the multivariate normal distribution using the evaluated mean and kernel function
        return gpytorch.distributions.MultivariateNormal(mean_x, cov_x) 
