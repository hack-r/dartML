import 'dart:math';

class DecisionTree {
  int maxDepth;
  List<double> thresholds = [];
  List<int> features = [];
  List<double> values = [];
  List<int> childrenLeft = [];
  List<int> childrenRight = [];
  List<List<double>> gradients;
  List<List<double>> hessians;

  DecisionTree(this.maxDepth, this.gradients, this.hessians) {
    int treeSize = pow(2, maxDepth + 1).toInt() - 1;
    thresholds = List<double>.filled(treeSize, 0.0);
    features = List<int>.filled(treeSize, 0);
    values = List<double>.filled(treeSize, 0.0);
    childrenLeft = List<int>.filled(treeSize, -1);
    childrenRight = List<int>.filled(treeSize, -1);
  }


  void fit(List<List<double>> X, List<double> weights) {
    int numSamples = X.length;

    List<int> indices = List<int>.generate(numSamples, (index) => index);
    indices.shuffle();

    List<List<double>> shuffledX = List<List<double>>.generate(numSamples, (i) => X[indices[i]]);
    List<double> shuffledWeights = List<double>.generate(numSamples, (i) => weights[indices[i]]);

    int nodeIndex = 0;
    List<int> sampleIndices = List<int>.generate(numSamples, (i) => i);

    splitNode(shuffledX, shuffledWeights, sampleIndices, nodeIndex);
  }

  void splitNode(List<List<double>> X, List<double> weights, List<int> sampleIndices, int nodeIndex) {
    List<double> y = List<double>.generate(sampleIndices.length, (i) => sampleIndices[i].toDouble());
    List<double> bestGradients = List<double>.filled(1, double.infinity);
    List<int> bestLeftIndices = [];
    List<int> bestRightIndices = [];
    int bestFeature = -1;
    double bestThreshold = 0;

    for (int feature = 0; feature < X[0].length; feature++) {
      List<double> featureValues = X.map((row) => row[feature]).toList();
      List<double> sortedValues = List<double>.from(featureValues);
      sortedValues.sort();

      for (int i = 0; i < sortedValues.length - 1; i++) {
        double threshold = (sortedValues[i] + sortedValues[i + 1]) / 2;
        List<int> leftIndices = [];
        List<int> rightIndices = [];

        for (int j = 0; j < sampleIndices.length; j++) {
          if (featureValues[sampleIndices[j]] <= threshold) {
            leftIndices.add(sampleIndices[j]);
          } else {
            rightIndices.add(sampleIndices[j]);
          }
        }

        // Calculate left and right gradients here (implement it according to your logic)
        List<double> leftGradients = [];
        List<double> rightGradients = [];

        double gain = calculateGain(weights, leftGradients, rightGradients);

        if (gain < bestGradients[0]) {
          bestGradients[0] = gain;
          bestLeftIndices = List<int>.from(leftIndices);
          bestRightIndices = List<int>.from(rightIndices);
          bestFeature = feature;
          bestThreshold = threshold;
        }
      }
    }

    if (bestFeature != -1 && bestGradients[0] != double.infinity) {
      features[nodeIndex] = bestFeature;
      thresholds[nodeIndex] = bestThreshold;

      int leftChildIndex = 2 * nodeIndex + 1;
      int rightChildIndex = 2 * nodeIndex + 2;

      childrenLeft[nodeIndex] = leftChildIndex;
      childrenRight[nodeIndex] = rightChildIndex;

      if (bestLeftIndices.isNotEmpty) {
        if (bestGradients[0] > 0 && getDepth(leftChildIndex) < maxDepth) {
          splitNode(X, weights, bestLeftIndices, leftChildIndex);
        } else {
          values[leftChildIndex] = calculateLeafValue(weights, bestLeftIndices)[0];
        }
      } else {
        childrenLeft[nodeIndex] = -1;
        values[leftChildIndex] = calculateLeafValue(weights, sampleIndices)[0];
      }

      if (bestRightIndices.isNotEmpty) {
        if (bestGradients[0] > 0 && getDepth(rightChildIndex) < maxDepth) {
          splitNode(X, weights, bestRightIndices, rightChildIndex);
        } else {
          values[rightChildIndex] = calculateLeafValue(weights, bestRightIndices)[0];
        }
      } else {
        childrenRight[nodeIndex] = -1;
        values[rightChildIndex] = calculateLeafValue(weights, sampleIndices)[0];
      }
    } else {
      childrenLeft[nodeIndex] = -1;
      childrenRight[nodeIndex] = -1;
      values[nodeIndex] = calculateLeafValue(weights, sampleIndices)[0];
    }
  }

  List<double> calculateLeafValue(List<double> weights, List<int> indices) {
    double numerator = sum([for (int index in indices) weights[index] * gradients[index][0]]);
    double denominator = sum([for (int index in indices) weights[index] * hessians[index][0]]);

    return [denominator != 0 ? numerator / denominator : 0.0];
  }

  int getDepth(int nodeIndex) {
    if (nodeIndex >= features.length) {
      return -1;
    }

    if (childrenLeft[nodeIndex] == -1 && childrenRight[nodeIndex] == -1) {
      return 0;
    }

    int leftDepth = getDepth(childrenLeft[nodeIndex]);
    int rightDepth = getDepth(childrenRight[nodeIndex]);

    return max(leftDepth, rightDepth) + 1;
  }

  double calculateGain(List<double> weights, List<double> leftGradients, List<double> rightGradients) {
    double numerator = pow(sum(leftGradients), 2).toDouble() + pow(sum(rightGradients), 2).toDouble();
    double denominator = sum([for (int i = 0; i < leftGradients.length; i++) weights[i] * leftGradients[i]]) +
        sum([for (int i = 0; i < rightGradients.length; i++) weights[i] * rightGradients[i]]);

    return denominator != 0 ? numerator / denominator : 0.0;
  }

  double sum(List<double> list) {
    return list.reduce((x, y) => x + y);
  }

  double predictRow(List<double> row) {
    int nodeIndex = 0;

    while (true) {
      int feature = features[nodeIndex];
      double threshold = thresholds[nodeIndex];

      if (feature != -1 && row[feature] <= threshold) {
        nodeIndex = childrenLeft[nodeIndex];
      } else if (feature != -1 && row[feature] > threshold) {
        nodeIndex = childrenRight[nodeIndex];
      } else {
        break;
      }
    }

    return values[nodeIndex];
  }
}