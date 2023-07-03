import 'dart:math';

class LogisticRegression {
  List<double> coefficients;

  LogisticRegression();

  void fit(List<List<double>> X, List<int> y, {int numIterations = 100, double learningRate = 0.1}) {
    int numSamples = X.length;
    int numFeatures = X[0].length;

    coefficients = List<double>.generate(numFeatures, (_) => 0.0);

    for (int iteration = 0; iteration < numIterations; iteration++) {
      for (int i = 0; i < numSamples; i++) {
        List<double> sample = X[i];
        int label = y[i];

        double predicted = predictProbability(sample);
        double error = label - predicted;

        for (int j = 0; j < numFeatures; j++) {
          coefficients[j] += learningRate * error * sample[j];
        }
      }
    }
  }

  double predictProbability(List<double> sample) {
    double linear = dotProduct(sample, coefficients);
    double probability = sigmoid(linear);
    return probability;
  }

  int predict(List<double> sample) {
    double probability = predictProbability(sample);
    if (probability >= 0.5) {
      return 1;
    } else {
      return 0;
    }
  }

  double dotProduct(List<double> a, List<double> b) {
    assert(a.length == b.length);

    double result = 0.0;
    for (int i = 0; i < a.length; i++) {
      result += a[i] * b[i];
    }
    return result;
  }

  double sigmoid(double x) {
    return 1.0 / (1.0 + exp(-x));
  }
}