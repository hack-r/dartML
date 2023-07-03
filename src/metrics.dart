class EvaluationMetrics {
  double accuracy(List<double> yTrue, List<double> yPred) {
    assert(yTrue.length == yPred.length);

    int correctPredictions = 0;
    for (int i = 0; i < yTrue.length; i++) {
      if ((yTrue[i] >= 0.5 && yPred[i] >= 0.5) || (yTrue[i] < 0.5 && yPred[i] < 0.5)) {
        correctPredictions++;
      }
    }
    return correctPredictions / yTrue.length;
  }
}

class CrossValidator {
  List<double> yTrue = [];
  List<double> yPred = [];

  void kFoldCrossValidation(dynamic model, List<List<double>> X, List<double> y, int k) {
    assert(X.length == y.length);
    int numSamples = X.length;

    List<int> indices = List.generate(numSamples, (index) => index);
    indices.shuffle();

    List<List<double>> shuffledX = List.generate(
        numSamples, (i) => X[indices[i]]);
    List<double> shuffledY = List.generate(numSamples, (i) => y[indices[i]]);

    int foldSize = (numSamples / k).floor();

    for (int i = 0; i < k; i++) {
      int startIdx = i * foldSize;
      int endIdx = (i + 1) * foldSize;
      if (i == k - 1) {
        endIdx = numSamples;
      }

      List<List<double>> trainX = List.from(shuffledX);
      List<double> trainY = List.from(shuffledY);
      List<List<double>> testX = shuffledX.sublist(startIdx, endIdx);
      List<double> testY = shuffledY.sublist(startIdx, endIdx);

      trainX.removeRange(startIdx, endIdx);
      trainY.removeRange(startIdx, endIdx);

      model.fit(trainX, trainY);

      List<double> foldPredictions = model.predict(testX);

      yTrue.addAll(testY);
      yPred.addAll(foldPredictions);
    }
  }

  void holdoutValidation(List<List<double>> X, List<double> y,
      double testSize) {
    assert(X.length == y.length);
    assert(testSize > 0 && testSize < 1);

    // holdout validation logic here
  }
}

class ConfusionMatrix {
  late List<List<int>> matrix;
  late int numClasses;

  ConfusionMatrix({required this.numClasses}) {
    matrix = List<List<int>>.generate(numClasses, (_) => List<int>.filled(numClasses, 0));
  }

  void addPrediction(int trueLabel, int predictedLabel) {
    matrix[trueLabel][predictedLabel]++;
  }

  int getTruePositives(int classIndex) {
    return matrix[classIndex][classIndex];
  }

  int getFalsePositives(int classIndex) {
    int falsePositives = 0;
    for (int i = 0; i < numClasses; i++) {
      if (i != classIndex) {
        falsePositives += matrix[i][classIndex];
      }
    }
    return falsePositives;
  }

  int getFalseNegatives(int classIndex) {
    int falseNegatives = 0;
    for (int i = 0; i < numClasses; i++) {
      if (i != classIndex) {
        falseNegatives += matrix[classIndex][i];
      }
    }
    return falseNegatives;
  }

  int getTrueNegatives(int classIndex) {
    int trueNegatives = 0;
    for (int i = 0; i < numClasses; i++) {
      for (int j = 0; j < numClasses; j++) {
        if (i != classIndex && j != classIndex) {
          trueNegatives += matrix[i][j];
        }
      }
    }
    return trueNegatives;
  }

  double getAccuracy() {
    int correctPredictions = 0;
    int totalPredictions = 0;

    for (int i = 0; i < numClasses; i++) {
      for (int j = 0; j < numClasses; j++) {
        totalPredictions += matrix[i][j];
        if (i == j) {
          correctPredictions += matrix[i][j];
        }
      }
    }

    if (totalPredictions == 0) {
      return 0.0;
    }

    return correctPredictions / totalPredictions.toDouble();
  }
}
