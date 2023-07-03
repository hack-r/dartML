class OLS {
  late List<double> coefficients;

  OLS();

  void fit(List<List<double>> X, List<double> y) {
    int numSamples = X.length;
    int numFeatures = X[0].length;

    List<List<double>> XTranspose = transpose(X);
    List<List<double>> XTX = matrixMultiply(XTranspose, X);
    List<List<double>> inverseXTX = matrixInverse(XTX);
    List<List<double>> XTY = matrixMultiply(XTranspose, columnVector(y));

    List<List<double>> coefficientsMatrix = matrixMultiply(inverseXTX, XTY);
    coefficients = coefficientsMatrix.map((row) => row[0]).toList();
  }

  List<double> predict(List<List<double>> X) {
    List<double> predictions = [];
    for (int i = 0; i < X.length; i++) {
      double prediction = dotProduct(X[i], coefficients);
      predictions.add(prediction);
    }
    return predictions;
  }

  List<List<double>> transpose(List<List<double>> matrix) {
    int numRows = matrix.length;
    int numCols = matrix[0].length;
    List<List<double>> transposed = List<List<double>>.generate(numCols, (_) => List<double>.generate(numRows, (_) => 0.0));
    for (int i = 0; i < numRows; i++) {
      for (int j = 0; j < numCols; j++) {
        transposed[j][i] = matrix[i][j];
      }
    }
    return transposed;
  }

  List<List<double>> matrixMultiply(List<List<double>> matrixA, List<List<double>> matrixB) {
    int numRowsA = matrixA.length;
    int numColsA = matrixA[0].length;
    int numRowsB = matrixB.length;
    int numColsB = matrixB[0].length;

    if (numColsA != numRowsB) {
      throw Exception("Incompatible matrix dimensions for multiplication");
    }

    List<List<double>> result = List<List<double>>.generate(numRowsA, (_) => List<double>.generate(numColsB, (_) => 0.0));
    for (int i = 0; i < numRowsA; i++) {
      for (int j = 0; j < numColsB; j++) {
        double sum = 0.0;
        for (int k = 0; k < numColsA; k++) {
          sum += matrixA[i][k] * matrixB[k][j];
        }
        result[i][j] = sum;
      }
    }
    return result;
  }

  List<List<double>> matrixInverse(List<List<double>> matrix) {
    // Implement matrix inverse algorithm
    // This is a simplified example and may not cover all cases
    double det = matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0];
    double invDet = 1.0 / det;

    List<List<double>> inverse = [
      [matrix[1][1] * invDet, -matrix[0][1] * invDet],
      [-matrix[1][0] * invDet, matrix[0][0] * invDet]
    ];

    return inverse;
  }

  List<List<double>> columnVector(List<double> list) {
    return list.map((value) => [value]).toList();
  }

  double dotProduct(List<double> a, List<double> b) {
    assert(a.length == b.length);

    double result = 0.0;
    for (int i = 0; i < a.length; i++) {
      result += a[i] * b[i];
    }
    return result;
  }
}

void main() {
  List<List<double>> X_train = [
    [1, 2],
    [3, 4],
    [5, 6]
  ];
  List<double> y_train = [3, 7, 11];

  OLS ols = OLS();
  ols.fit(X_train, y_train);

  List<List<double>> X_test = [
    [2, 3],
    [4, 5]
  ];
  List<double> predictions = ols.predict(X_test);

  print('Predicted values: $predictions');
}
