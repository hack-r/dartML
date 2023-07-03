import 'dart:math';

class SOM {
  late List<List<double>> weightMatrix;
  int inputSize;
  int mapSize;
  int numIterations;
  double learningRate;
  double neighborhoodRadius;

  SOM({required this.inputSize, required this.mapSize, required this.numIterations, required this.learningRate, required this.neighborhoodRadius}) {
    weightMatrix = List<List<double>>.generate(mapSize, (_) => List<double>.generate(inputSize, (_) => randomWeight()));
  }

  void train(List<List<double>> input) {
    for (int iteration = 0; iteration < numIterations; iteration++) {
      double currentLearningRate = learningRate * exp(-iteration / numIterations);
      double currentRadius = neighborhoodRadius * exp(-iteration / numIterations);

      for (List<double> sample in input) {
        int bestMatchingUnitIndex = findBestMatchingUnitIndex(sample);
        int bestMatchingUnitRow = bestMatchingUnitIndex ~/ mapSize;
        int bestMatchingUnitColumn = bestMatchingUnitIndex % mapSize;

        for (int i = 0; i < mapSize; i++) {
          for (int j = 0; j < mapSize; j++) {
            double distance = sqrt(pow(i - bestMatchingUnitRow, 2) + pow(j - bestMatchingUnitColumn, 2));

            if (distance <= currentRadius) {
              double influence = exp(-pow(distance, 2) / (2 * pow(currentRadius, 2)));

              for (int k = 0; k < inputSize; k++) {
                weightMatrix[i][j] += currentLearningRate * influence * (sample[k] - weightMatrix[i][j]);
              }
            }
          }
        }
      }
    }
  }

  int findBestMatchingUnitIndex(List<double> sample) {
    double bestDistance = double.infinity;
    int bestIndex = -1;

    for (int i = 0; i < mapSize; i++) {
      for (int j = 0; j < mapSize; j++) {
        double distance = calculateDistance(sample, weightMatrix[i]);

        if (distance < bestDistance) {
          bestDistance = distance;
          bestIndex = i * mapSize + j;
        }
      }
    }

    return bestIndex;
  }

  double calculateDistance(List<double> a, List<double> b) {
    assert(a.length == b.length);

    double sumOfSquares = 0.0;
    for (int i = 0; i < a.length; i++) {
      sumOfSquares += pow(a[i] - b[i], 2);
    }
    return sqrt(sumOfSquares);
  }

  double randomWeight() {
    // Generates a random weight between 0 and 1
    return Random().nextDouble();
  }

  List<List<int>> getClusterAssignments(List<List<double>> input) {
    List<List<int>> clusterAssignments = List<List<int>>.generate(mapSize, (_) => List<int>.generate(mapSize, (_) => -1));

    for (int i = 0; i < input.length; i++) {
      List<double> sample = input[i];
      int bestMatchingUnitIndex = findBestMatchingUnitIndex(sample);
      int bestMatchingUnitRow = bestMatchingUnitIndex ~/ mapSize;
      int bestMatchingUnitColumn = bestMatchingUnitIndex % mapSize;
      clusterAssignments[bestMatchingUnitRow][bestMatchingUnitColumn] = i;
    }

    return clusterAssignments;
  }
}