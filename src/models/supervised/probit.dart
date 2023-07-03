import 'dart:math';

class ProbitRegression {
  late List<double> beta;

  ProbitRegression();

  void fit(List<List<double>> X, List<double> y, {int maxIter = 100, double epsilon = 1e-6}) {
    int numSamples = X.length;
    int numFeatures = X[0].length;
    beta = List.filled(numFeatures, 0);

    // Define the Probit link function
    double probitLinkFunction(double x) {
      return 0.5 * (1 + tanh(x / sqrt(2)));
    }

    int iter = 0;
    while (iter < maxIter) {
      // Compute the predicted probabilities P(X) for each observation
      List<double> predictedProbs = X.map((row) => probitLinkFunction(dotProduct(row, beta))).toList();

      // Compute the vector of differences between predicted and observed probabilities
      List<double> diff = List.generate(numSamples, (i) => y[i] - predictedProbs[i]);

      // Compute the diagonal matrix of weights based on the predicted probabilities
      List<List<double>> weights = List.generate(numSamples, (i) => List.filled(numSamples, 0.0));
      for (int i = 0; i < numSamples; i++) {
        weights[i][i] = predictedProbs[i] * (1 - predictedProbs[i]);
      }

      // Compute the Hessian matrix
      List<List<double>> XWeighted = List.generate(numSamples, (i) => List.filled(numFeatures, 0.0));
      for (int i = 0; i < numSamples; i++) {
        for (int j = 0; j < numFeatures; j++) {
          XWeighted[i][j] = weights[i][i] * X[i][j];
        }
      }
      List<List<double>> hessian = List.generate(numFeatures, (j) => List.filled(numFeatures, 0.0));
      for (int j = 0; j < numFeatures; j++) {
        for (int k = 0; k < numFeatures; k++) {
          hessian[j][k] = -sum([for (int i = 0; i < numSamples; i++) XWeighted[i][j] * X[i][k]]);
        }
      }

      // Compute the gradient vector
      List<double> gradient = List.generate(numFeatures, (j) => -sum([for (int i = 0; i < numSamples; i++) weights[i][i] * diff[i] * X[i][j]]));

      // Compute the updated β using the Newton-Raphson update formula
      List<double> betaUpdate = solveLinearSystem(hessian, gradient);

      // Check for convergence
      if (norm(betaUpdate) < epsilon) {
        break;
      }

      // Update the β coefficients
      for (int j = 0; j < numFeatures; j++) {
        beta[j] += betaUpdate[j];
      }

      iter++;
    }

    // Output the estimated coefficients
    print('Estimated coefficients: $beta');
  }

  List<double> predict(List<List<double>> X) {
    // Define the Probit link function
    double probitLinkFunction(double x) {
      return 0.5 * (1 + tanh(x / sqrt(2)));
    }

    return X.map((row) => probitLinkFunction(dotProduct(row, beta))).toList();
  }

  double dotProduct(List<double> a, List<double> b) {
    assert(a.length == b.length);

    double result = 0;
    for (int i = 0; i < a.length; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  double norm(List<double> a) {
    double sumOfSquares = a.map((value) => value * value).reduce((x, y) => x + y);
    return sqrt(sumOfSquares);
  }

  List<double> solveLinearSystem(List<List<double>> A, List<double> b) {
    assert(A.length == A[0].length);
    assert(A.length == b.length);

    int n = A.length;
    List<List<double>> augmentedMatrix = List.generate(n, (i) => List.filled(n + 1, 0.0));

    for (int i = 0; i < n; i++) {
      augmentedMatrix[i] = [...A[i], b[i]];
    }

    for (int i = 0; i < n; i++) {
      if (augmentedMatrix[i][i] == 0) {
        for (int j = i + 1; j < n; j++) {
          if (augmentedMatrix[j][i] != 0) {
            swapRows(augmentedMatrix, i, j);
            break;
          }
        }
      }

      double pivot = augmentedMatrix[i][i];
      for (int j = i; j <= n; j++) {
        augmentedMatrix[i][j] /= pivot;
      }

      for (int j = 0; j < n; j++) {
        if (j != i) {
          double factor = augmentedMatrix[j][i];
          for (int k = i; k <= n; k++) {
            augmentedMatrix[j][k] -= factor * augmentedMatrix[i][k];
          }
        }
      }
    }

    return augmentedMatrix.map((row) => row[n]).toList();
  }

  void swapRows(List<List<double>> matrix, int i, int j) {
    List<double> temp = matrix[i];
    matrix[i] = matrix[j];
    matrix[j] = temp;
  }

  double tanh(double x) {
    return (exp(x) - exp(-x)) / (exp(x) + exp(-x));
  }

  double sum(List<double> list) {
    return list.reduce((x, y) => x + y);
  }
}