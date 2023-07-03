import 'dart:math';

class PropensityScoreMatching {
  List<double> propensityScores = [];

  PropensityScoreMatching();

  void calculatePropensityScores(List<List<double>> X, List<int> treatment) {
    int numSamples = X.length;
    List<double> probabilities = List<double>.filled(numSamples, 0.0);
    propensityScores = List<double>.filled(numSamples, 0.0);

    // Estimate propensity scores using logistic regression or any other method
    // Here, we assume a random probability between 0 and 1 as an example
    for (int i = 0; i < numSamples; i++) {
      probabilities[i] = Random().nextDouble();
    }

    // Normalize the propensity scores to sum up to 1
    double sumProbabilities = probabilities.reduce((a, b) => a + b);
    for (int i = 0; i < numSamples; i++) {
      propensityScores[i] = probabilities[i] / sumProbabilities;
    }
  }

  List<List<double>> performMatching(List<List<double>> X, List<int> treatment, {int numMatches = 1}) {
    int numSamples = X.length;
    List<List<double>> matchedData = [];

    // Perform matching for each treated sample
    for (int i = 0; i < numSamples; i++) {
      if (treatment[i] == 1) {
        List<double> sample = X[i];
        double propensityScore = propensityScores[i];

        // Find the nearest matches based on propensity score
        List<List<double>> distancesAndIndices = [];
        for (int j = 0; j < numSamples; j++) {
          if (treatment[j] == 0) {
            double distance = (propensityScores[j] - propensityScore).abs();
            distancesAndIndices.add([distance, j.toDouble()]);
          }
        }

        // Sort based on the distance and select the nearest matches based on the provided number of matches
        distancesAndIndices.sort((a, b) => a[0].compareTo(b[0]));

        if (distancesAndIndices.length > numMatches) {
          distancesAndIndices = distancesAndIndices.sublist(0, numMatches);
        }

        // Add the matched samples to the matchedData list
        for (List<double> distanceAndIndex in distancesAndIndices) {
          int matchIndex = distanceAndIndex[1].toInt();
          matchedData.add([...sample, ...X[matchIndex]]);
        }
      }
    }

    return matchedData;
  }
}
