import 'dart:math';
import 'decision_tree.dart';

class GradientBoostingClassifier {
  List<double> yTrue;
  List<double> yPred;
  List<DecisionTree> trees;
  List<double> learningRates;

  GradientBoostingClassifier({int numTrees = 100}) {
    trees = List<DecisionTree>.filled(numTrees, DecisionTree());
    learningRates = List<double>.filled(numTrees, 0.1);
  }

  void fit(List<List<double>> X, List<double> y) {
    assert(X.length == y.length);
    yTrue = List.from(y);
    yPred = List.filled(y.length, 0.0);

    for (int i = 0; i < trees.length; i++) {
      List<double> residuals = calculateResiduals(yTrue, yPred);

      trees[i].fit(X, residuals);
      List<double> treePredictions = trees[i].predict(X);

      for (int j = 0; j < yPred.length; j++) {
        yPred[j] += learningRates[i] * treePredictions[j];
      }
    }
  }

  List<double> predict(List<List<double>> X) {
    List<double> predictions = List.filled(X.length, 0.0);

    for (int i = 0; i < trees.length; i++) {
      List<double> treePredictions = trees[i].predict(X);

      for (int j = 0; j < predictions.length; j++) {
        predictions[j] += learningRates[i] * treePredictions[j];
      }
    }

    return predictions.map((prediction) => sigmoid(prediction)).toList();
  }

  List<double> calculateResiduals(List<double> yTrue, List<double> yPred) {
    assert(yTrue.length == yPred.length);

    return List.generate(yTrue.length, (i) => yTrue[i] - sigmoid(yPred[i]));
  }

  double sigmoid(double x) {
    return 1 / (1 + exp(-x));
  }
}