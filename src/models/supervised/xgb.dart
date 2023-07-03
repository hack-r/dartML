import 'dart:math';
import 'decision_tree.dart';

class XGBoostClassifier {
  List<double> yTrue;
  List<double> yPred;
  List<double> weights;
  List<double> baseScores;
  List<List<double>> gradients;
  List<List<double>> hessians;
  List<List<double>> trees;
  List<double> learningRates;

  XGBoostClassifier({int numIterations = 100, double learningRate = 0.1}) {
    trees = [];
    learningRates = List<double>.filled(numIterations, learningRate);
  }

  void fit(List<List<double>> X, List<double> y) {
    assert(X.length == y.length);
    yTrue = List.from(y);
    yPred = List.filled(y.length, 0.0);
    weights = List.filled(y.length, 1.0);
    baseScores = List.filled(y.length, 0.0);
    gradients = List<List<double>>.filled(y.length, List.filled(1, 0.0));
    hessians = List<List<double>>.filled(y.length, List.filled(1, 0.0));

    for (int iter = 0; iter < trees.length; iter++) {
      calculateGradientsAndHessians();

      List<double> treePredictions = List<double>.filled(y.length, 0.0);
      List<double> treeWeights = List<double>.filled(y.length, 0.0);

      for (int i = 0; i < y.length; i++) {
        treeWeights[i] = weights[i] * sqrt(hessians[i][0]);
      }

      DecisionTree tree = DecisionTree();
      tree.fit(X, treeWeights);

      for (int i = 0; i < y.length; i++) {
        treePredictions[i] = tree.predictRow(X[i]);
      }

      trees.add(treePredictions);
      updatePredictions(iter);
    }
  }

  List<double> predict(List<List<double>> X) {
    List<double> predictions = List<double>.filled(X.length, 0.0);

    for (int i = 0; i < trees.length; i++) {
      List<double> treePredictions = trees[i];

      for (int j = 0; j < predictions.length; j++) {
        predictions[j] += learningRates[i] * treePredictions[j];
      }
    }

    return predictions.map((prediction) => sigmoid(prediction)).toList();
  }

  void calculateGradientsAndHessians() {
    for (int i = 0; i < yTrue.length; i++) {
      gradients[i][0] = yTrue[i] - sigmoid(baseScores[i] + yPred[i]);
      hessians[i][0] = sigmoid(baseScores[i] + yPred[i]) * (1 - sigmoid(baseScores[i] + yPred[i]));
    }
  }

  void updatePredictions(int iter) {
    for (int i = 0; i < yPred.length; i++) {
      yPred[i] += learningRates[iter] * trees[iter][i];
    }
  }

  double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }
}
