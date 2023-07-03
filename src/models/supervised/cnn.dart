import 'dart:math';

double sigmoid(double x) {
  return 1 / (1 + exp(-x));
}

double relu(double x) {
  return max(0, x);
}

class ConvolutionalLayer {
  List<List<List<double>>> filters = [];
  List<double> biases = [];

  ConvolutionalLayer({int filterSize = 3, int numFilters = 1}) {
    filters = List<List<List<double>>>.generate(
      numFilters,
          (_) => List<List<double>>.generate(
        filterSize,
            (_) => List<double>.generate(filterSize, (_) => heInitialization()),
      ),
    );
    biases = List<double>.generate(numFilters, (_) => heInitialization());
  }

  List<List<double>> apply(List<List<double>> input) {
    int inputSize = input.length;
    int filterSize = filters[0].length;
    int outputSize = inputSize - filterSize + 1;

    List<List<double>> output = List<List<double>>.generate(
      outputSize,
          (_) => List<double>.generate(outputSize, (_) => 0.0),
    );

    for (int filterIndex = 0; filterIndex < filters.length; filterIndex++) {
      for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
          double activation = 0.0;
          for (int m = 0; m < filterSize; m++) {
            for (int n = 0; n < filterSize; n++) {
              activation += input[i + m][j + n] * filters[filterIndex][m][n];
            }
          }
          activation += biases[filterIndex];
          output[i][j] = relu(activation);
        }
      }
    }

    return output;
  }

  double heInitialization() {
    double standardDeviation = sqrt(2 / (filters.length * filters[0].length * filters[0][0].length));
    return Random().nextDouble() * standardDeviation;
  }

  void backpropagation(List<List<double>> input, List<List<double>> outputGradients, List<List<List<double>>> filterGradients, List<double> biasGradients) {
    int inputSize = input.length;
    int filterSize = filters[0].length;
    int outputSize = inputSize - filterSize + 1;

    for (int filterIndex = 0; filterIndex < filters.length; filterIndex++) {
      for (int i = 0; i < outputSize; i++) {
        for (int j = 0; j < outputSize; j++) {
          double outputGradient = outputGradients[i][j];
          for (int m = 0; m < filterSize; m++) {
            for (int n = 0; n < filterSize; n++) {
              filterGradients[filterIndex][m][n] += input[i + m][j + n] * (outputGradient > 0 ? 1 : 0);
            }
          }
          biasGradients[filterIndex] += outputGradient > 0 ? 1 : 0;
        }
      }
    }
  }

  void updateParameters(double learningRate, int batchSize, List<List<List<double>>> filterGradients, List<double> biasGradients) {
    int numFilters = filters.length;
    int filterSize = filters[0].length;
    int numChannels = filters[0][0].length;

    for (int filterIndex = 0; filterIndex < numFilters; filterIndex++) {
      for (int i = 0; i < filterSize; i++) {
        for (int j = 0; j < filterSize; j++) {
          for (int k = 0; k < numChannels; k++) {
            filters[filterIndex][i][j] -= (filterGradients[filterIndex][i][j] / batchSize) * learningRate;
          }
        }
      }
      biases[filterIndex] -= (biasGradients[filterIndex] / batchSize) * learningRate;
    }
  }
}

class MaxPoolingLayer {
  int poolSize;

  MaxPoolingLayer({this.poolSize = 2});

  List<List<double>> apply(List<List<double>> input) {
    int inputSize = input.length;
    int outputSize = (inputSize / poolSize).floor();

    List<List<double>> output = List<List<double>>.generate(
      outputSize,
          (_) => List<double>.generate(outputSize, (_) => 0.0),
    );

    for (int i = 0; i < outputSize; i++) {
      for (int j = 0; j < outputSize; j++) {
        double maxVal = double.negativeInfinity;
        for (int m = 0; m < poolSize; m++) {
          for (int n = 0; n < poolSize; n++) {
            maxVal = max(maxVal, input[i * poolSize + m][j * poolSize + n]);
          }
        }
        output[i][j] = maxVal;
      }
    }

    return output;
  }

  void backpropagation(List<List<double>> input, List<List<double>> outputGradients, List<List<double>> inputGradients) {
    int inputSize = input.length;
    int outputSize = (inputSize / poolSize).floor();

    for (int i = 0; i < outputSize; i++) {
      for (int j = 0; j < outputSize; j++) {
        double outputGradient = outputGradients[i][j];
        for (int m = 0; m < poolSize; m++) {
          for (int n = 0; n < poolSize; n++) {
            int maxIndexI = i * poolSize + m;
            int maxIndexJ = j * poolSize + n;
            inputGradients[maxIndexI][maxIndexJ] = input[maxIndexI][maxIndexJ] == outputGradient ? 1 : 0;
          }
        }
      }
    }
  }
}

class FullyConnectedLayer {
  List<List<double>> weights = [];
  List<double> biases = [];
  List<List<double>> inputGradients = [];
  List<List<double>> weightGradients = [];
  List<double> biasGradients = [];
  int inputSize;
  int outputSize;

  FullyConnectedLayer({required this.inputSize, required this.outputSize}) {
    weights = List<List<double>>.generate(
      outputSize,
          (_) => List<double>.generate(inputSize, (_) => heInitialization()),
    );
    biases = List<double>.generate(outputSize, (_) => heInitialization());
  }

  List<double> apply(List<double> input) {
    List<double> output = List<double>.generate(outputSize, (_) => 0.0);

    for (int i = 0; i < outputSize; i++) {
      double activation = 0.0;
      for (int j = 0; j < inputSize; j++) {
        activation += input[j] * weights[i][j];
      }
      activation += biases[i];
      output[i] = relu(activation);
    }

    return output;
  }

  double heInitialization() {
    double standardDeviation = sqrt(2 / inputSize);
    return Random().nextDouble() * standardDeviation;
  }

  void backpropagation(List<double> input, List<double> outputGradients) {
    for (int i = 0; i < outputSize; i++) {
      double outputGradient = outputGradients[i];
      double reluGradient = output[i] > 0 ? 1 : 0;
      for (int j = 0; j < inputSize; j++) {
        weightGradients[i][j] += input[j] * outputGradient * reluGradient;
        inputGradients[j] += weights[i][j] * outputGradient * reluGradient;
      }
      biasGradients[i] += outputGradient * reluGradient;
    }
  }

  void updateParameters(double learningRate, int batchSize) {
    for (int i = 0; i < outputSize; i++) {
      for (int j = 0; j < inputSize; j++) {
        weights[i][j] -= (weightGradients[i][j] / batchSize) * learningRate;
      }
      biases[i] -= (biasGradients[i] / batchSize) * learningRate;
    }
  }
}

class CNN {
  ConvolutionalLayer convolutionalLayer;
  MaxPoolingLayer maxPoolingLayer;
  FullyConnectedLayer fullyConnectedLayer;

  CNN({
    required int inputSize,
    required int numFilters,
    required int filterSize,
    required int poolSize,
    required int outputSize,
  })   : convolutionalLayer = ConvolutionalLayer(filterSize: filterSize, numFilters: numFilters),
        maxPoolingLayer = MaxPoolingLayer(poolSize: poolSize),
        fullyConnectedLayer = FullyConnectedLayer(inputSize: inputSize, outputSize: outputSize);

  int calculateHiddenSize(int inputSize, int filterSize, int poolSize) {
    // Calculate the size of the output from the MaxPooling layer
    int convolvedSize = inputSize - filterSize + 1;
    int pooledSize = (convolvedSize / poolSize).floor();
    return pooledSize * pooledSize;
  }

  List<double> predict(List<List<double>> input) {
    List<List<double>> convolved = convolutionalLayer.apply(input);
    List<List<double>> pooled = maxPoolingLayer.apply(convolved);
    List<double> flattened = pooled.expand((row) => row).toList();
    return fullyConnectedLayer.apply(flattened);
  }

  void train(List<List<double>> input, List<double> target, double learningRate, int batchSize, int numEpochs) {
    for (int epoch = 0; epoch < numEpochs; epoch++) {
      List<double> lossGradients = List<double>.filled(target.length, 0.0);
      int numBatches = input.length ~/ batchSize;

      for (int batchIndex = 0; batchIndex < numBatches; batchIndex++) {
        int startIndex = batchIndex * batchSize;
        int endIndex = (batchIndex + 1) * batchSize;
        List<List<double>> batchInput = input.sublist(startIndex, endIndex);
        List<double> batchTarget = target.sublist(startIndex, endIndex);

        // Forward pass
        List<List<double>> convolvedBatch = [];
        List<List<double>> pooledBatch = [];
        List<List<double>> flattenedBatch = [];

        for (int i = 0; i < batchSize; i++) {
          List<List<double>> convolved = convolutionalLayer.apply(batchInput[i]);
          List<List<double>> pooled = maxPoolingLayer.apply(convolved);
          List<double> flattened = pooled.expand((row) => row).toList();
          convolvedBatch.add(convolved);
          pooledBatch.add(pooled);
          flattenedBatch.add(flattened);
        }

        List<double> outputBatch = fullyConnectedLayer.apply(flattenedBatch);

        // Backpropagation
        List<double> outputGradients = [];
        List<List<double>> flattenedGradients = [];
        List<List<double>> pooledGradients = [];
        List<List<double>> convolvedGradients = [];

        for (int i = 0; i < batchSize; i++) {
          List<double> outputGradient = [];
          List<double> flattenedGradient = List<double>.filled(flattenedBatch[i].length, 0.0);
          List<List<double>> pooledGradient = List<List<double>>.generate(
            pooledBatch[i].length,
                (_) => List<double>.filled(pooledBatch[i][0].length, 0.0),
          );
          List<List<double>> convolvedGradient = List<List<double>>.generate(
            convolvedBatch[i].length,
                (_) => List<double>.filled(convolvedBatch[i][0].length, 0.0),
          );

          for (int j = 0; j < outputBatch.length; j++) {
            double lossGradient = outputBatch[j] - batchTarget[i];
            outputGradients.add(lossGradient);
            lossGradients[j] += lossGradient / batchSize;
            double reluGradient = outputBatch[j] > 0 ? 1 : 0;

            for (int k = 0; k < flattenedBatch[i].length; k++) {
              flattenedGradient[k] += fullyConnectedLayer.weights[j][k] * lossGradient * reluGradient;
            }

            for (int m = 0; m < pooledBatch[i].length; m++) {
              for (int n = 0; n < pooledBatch[i][0].length; n++) {
                pooledGradient[m][n] = flattenedGradient[m * pooledBatch[i][0].length + n];
              }
            }

            for (int p = 0; p < convolvedBatch[i].length; p++) {
              for (int q = 0; q < convolvedBatch[i][0].length; q++) {
                double convolvedValue = convolvedBatch[i][p][q];
                convolvedGradient[p][q] = pooledGradient[p ~/ maxPoolingLayer.poolSize][q ~/ maxPoolingLayer.poolSize] * (convolvedValue > 0 ? 1 : 0);
              }
            }
          }

          outputGradients.add(outputGradient);
          flattenedGradients.add(flattenedGradient);
          pooledGradients.add(pooledGradient);
          convolvedGradients.add(convolvedGradient);
        }

        // Update parameters
        fullyConnectedLayer.updateParameters(learningRate, batchSize);
        for (int i = 0; i < batchSize; i++) {
          fullyConnectedLayer.backpropagation(flattenedBatch[i], outputGradients[i]);
        }

        for (int i = 0; i < batchSize; i++) {
          maxPoolingLayer.backpropagation(convolvedBatch[i], pooledGradients[i], convolvedGradients[i]);
          convolutionalLayer.backpropagation(batchInput[i], convolvedGradients[i], convolutionalLayer.filters, convolutionalLayer.biases);
        }
      }

      // Print loss for each epoch
      double epochLoss = lossGradients.fold(0.0, (a, b) => a + b);
      print('Epoch ${epoch + 1}: Loss = $epochLoss');
    }
  }
}
