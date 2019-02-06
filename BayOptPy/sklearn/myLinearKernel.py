import numpy as np

from sklearn.gaussian_process.kernels import Kernel, Hyperparameter, _check_length_scale


# Use the code availible in a pull request that has not been merged to sklearn yet:
# https://github.com/scikit-learn/scikit-learn/pull/10597/files


class LinearKernel(Kernel):
    """Linear kernel.
     The Linear kernel is non-stationary and can be obtained from Bayesian
    linear regression by shifting the origin by c. The parameters of the Linear
    kernel are about specifying the origin. The kernel is given by:
     k(x_i, x_j) = ((x_i - c) \cdot (x_j - c))
     The Linear kernel can be combined with other kernels, more commonly with
    periodic kernels.
     Parameters
    ----------
    c : float or array with shape (n_features,), default: 0.0
        The offset for the origin. If a float, the same offset is used
        throughout all dimensions. If an array, each dimension of c defines the
        offset of the respective feature dimension.
     c_bounds : pair of floats >= 0, default: (1e-5, 1e5)
        The lower and upper bound on c
    """
    def __init__(self, c=0.0, c_bounds=(1e-5, 1e5)):
        self.c = c
        self.c_bounds = c_bounds

    @property
    def non_uniform_offset(self):
        return np.iterable(self.c) and len(self.c) > 1
    @property
    def hyperparameter_c(self):
        if self.non_uniform_offset:
            return Hyperparameter("c", "numeric", self.c_bounds, len(self.c))
        else:
            return Hyperparameter("c", "numeric", self.c_bounds)

    def __call__(self, X, Y=None, eval_gradient=False):
       """Return the kernel k(X, Y) and optionally its gradient.
        Parameters
       ----------
       X : array, shape (n_samples_X, n_features)
           Left argument of the returned kernel k(X, Y)
        Y : array, shape (n_samples_Y, n_features), (optional, default=None)
           Right argument of the returned kernel k(X, Y). If None, k(X, X)
           if evaluated instead.
        eval_gradient : bool (optional, default=False)
           Determines whether the gradient with respect to the kernel
           hyperparameter is determined. Only supported when Y is None.
        Returns
       -------
       K : array, shape (n_samples_X, n_samples_Y)
           Kernel k(X, Y)
        K_gradient : array (opt.), shape (n_samples_X, n_samples_X, n_dims)
           The gradient of the kernel k(X, X) with respect to the
           hyperparameter of the kernel. Only returned when eval_gradient
           is True.
       """
       X = np.atleast_2d(X)
       self.c = _check_length_scale(X, self.c)
       if self.c is 0:
           Xsub = X
       else:
           Xsub = X - self.c
       if Y is None:
          K = np.inner(Xsub, Xsub)
       else:
           if eval_gradient:
               raise ValueError(
                   "Gradient can only be evaluated when Y is None.")
           K = np.inner(Xsub, Y - self.c)
       if eval_gradient:
          if self.hyperparameter_c.fixed:
              c_gradient = np.empty((X.shape[0], X.shape[0], 0))
          else:
              if not self.non_uniform_offset:
                  gradient_mat = np.inner(np.ones(X.shape), Xsub)
                  c_gradient = - self.c * (gradient_mat + gradient_mat.T)
                  c_gradient = c_gradient[:, :, np.newaxis]
              else:
                  c_gradient = []
                  for i, c in enumerate(self.c):
                      gradient_mat = np.vstack(
                          [Xsub[:, i]] * X.shape[0])
                      c_gradient.append(c * (gradient_mat + gradient_mat.T))
                  c_gradient = -np.array(c_gradient)
                  c_gradient = np.rollaxis(c_gradient, 0, 3)
          return K, c_gradient
       else:
           return K

    def diag(self, X):
       """Returns the diagonal of the kernel k(X, X).
        The result of this method is identical to np.diag(self(X)); however,
       it can be evaluated more efficiently since only the diagonal is
       evaluated.
        Parameters
       ----------
       X : array, shape (n_samples_X, n_features)
           Left argument of the returned kernel k(X, Y)
        Returns
       -------
       K_diag : array, shape (n_samples_X,)
           Diagonal of kernel k(X, X)
       """
       return np.einsum('ij,ij->i', X - self.c, X - self.c)

    def is_stationary(self):
       """Returns whether the kernel is stationary."""
       return False

    def __repr__(self):
       if self.non_uniform_offset:
           return "{0}(c=[{1}])".format(
               self.__class__.__name__,
               ", ".join(map("{0:.3g}".format, self.c)))
       else:
           return "{0}(c={1:.3g})".format(
               self.__class__.__name__, np.ravel(self.c)[0])