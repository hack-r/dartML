import 'dart:math';

class Anova {
  double calculateFValue(List<List<double>> groups) {
    int numGroups = groups.length;
    int numSamplesPerGroup = groups[0].length;
    int totalNumSamples = numGroups * numSamplesPerGroup;

    double grandMean = calculateGrandMean(groups);
    double ssBetween = calculateSSBetween(groups, grandMean);
    double ssWithin = calculateSSWithin(groups);
    double msBetween = ssBetween / (numGroups - 1);
    double msWithin = ssWithin / (totalNumSamples - numGroups);

    return msBetween / msWithin;
  }

  double calculateGrandMean(List<List<double>> groups) {
    double sum = 0.0;
    int totalNumSamples = 0;

    for (List<double> group in groups) {
      sum += group.reduce((a, b) => a + b);
      totalNumSamples += group.length;
    }

    return sum / totalNumSamples;
  }

  double calculateSSBetween(List<List<double>> groups, double grandMean) {
    double ssBetween = 0.0;

    for (List<double> group in groups) {
      double groupMean = group.reduce((a, b) => a + b) / group.length;
      ssBetween += group.length * pow(groupMean - grandMean, 2);
    }

    return ssBetween;
  }

  double calculateSSWithin(List<List<double>> groups) {
    double ssWithin = 0.0;

    for (List<double> group in groups) {
      double groupMean = group.reduce((a, b) => a + b) / group.length;
      double sumOfSquares = 0.0;
      for (double value in group) {
        sumOfSquares += pow(value - groupMean, 2);
      }
      ssWithin += sumOfSquares;
    }

    return ssWithin;
  }
}