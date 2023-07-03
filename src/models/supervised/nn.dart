import 'dart:math';

class NeuralNetwork {
  List<List<double>> weights;
  List<double> biases;
  List<int> layers;

  NeuralNetwork(List<int> layerSizes) {
    layers = List<int>.from(layerSizes);
    weights = [];
    biases = [];

    for (int i = 1; i < layers.length; i++) {
      int prevLayerSize = layers[i - 1];
      int currentLayerSize = layers[i];

      List<List<double>> layerWeights = List<List<double>>.generate(currentLayerSize, (_) => List<double>.generate(prevLayerSize, (_) => randomWeight()));
      weights.add(layerWeights);

      List<double> layerBiases = List<double>.generate(currentLayerSize, (_) => randomWeight());
      biases.add(layerBiases);
    }
  }

  double randomWeight() {
    // Generates a random weight between -1 and 1
    return Random().nextDouble() * 2 - 1;
  }

  List<double> predict(List<double> input) {
    List<double> activations = List<double>.from(input);

    for (int i = 0; i < weights.length; i++) {
      List<List<double>> layerWeights = weights[i];
      List<double> layerBiases = biases[i];
      List<double> newActivations = [];

      for (int j = 0; j < layerWeights.length; j++) {
        double activation = dotProduct(layerWeights[j], activations) + layerBiases[j];
        newActivations.add(sigmoid(activation));
      }

      activations = List<double>.from(newActivations);
    }

    return activations;
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
    return 1 / (1 + exp(-x));
  }
}