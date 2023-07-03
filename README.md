# dartML

dartML is a Dart library that provides various machine learning algorithms and utilities for data analysis and model building.

The function implementations are basic compared to scikit-learn, caret, etc. but, as far as I can tell, this is the first Machine Learning library for Dart.

## Status

Under construction!! This library is still in development and until published on pub.dev please don't assume that functions are ready for use.

Currently, the following are (more or less) working:
 - ANOVA
 - Decision Trees
 - Metrics and Helper Functions
 - Probit
 - OLS
 - Self-Organizing Maps

## Features

- Implementation of popular machine learning algorithms including neural networks (CNN, feed forward, etc), gradient boosting, eXtreme gradient boosting (XGB), Logistic Regression, Propensity Score Matching, and more.
- Metrics for evaluating the performance of machine learning models such as Accuracy, Confusion Matrix, F-value, and more.
- Reinforcement Learning
- Unsupervised Learning

## Installation

Add this library as a dependency in your Dart project's `pubspec.yaml` file:

```yaml
dependencies:
  dartML: ^1.0.0
```

Then, run the following command to fetch the package:

$ dart pub get

## Usage

Import the library into your Dart file:

import 'package:dartML/dartML.dart';


### Decision Trees

```dart
import 'package:dartML/models/supervised/logistic_regression.dart';

void main() {
  // Load your dataset and preprocess it as needed

  List<List<double>> X = ...; // Features
  List<int> y = ...; // Labels

  // Create a LogisticRegression instance
  LogisticRegression logit = LogisticRegression();

  // Train the model
  logit.fit(X, y);

  // Make predictions
  List<double> sample = ...; // Sample to predict
  int predictedLabel = logit.predict(sample);

  print('Predicted label: $predictedLabel');
}
```
```dart
import 'package:dartML/models/supervised/decision_tree.dart';

void main() {
// Load your dataset and preprocess it as needed

List<List<double>> X = ...; // Features
List<int> y = ...; // Labels

// Create a DecisionTree instance
DecisionTree decisionTree = DecisionTree();

// Train the model
decisionTree.fit(X, y);

// Make predictions
List<double> sample = ...; // Sample to predict
int predictedLabel = decisionTree.predict(sample);

print('Predicted label: $predictedLabel');
}
```
### Gradient Boosting

```dart
import 'package:dartML/models/supervised/gradient_boosting.dart';

void main() {
  // Load your dataset and preprocess it as needed

  List<List<double>> X = ...; // Features
  List<int> y = ...; // Labels

  // Create a GradientBoosting instance
  GradientBoosting gb = GradientBoosting();

  // Train the model
  gb.fit(X, y);

  // Make predictions
  List<double> sample = ...; // Sample to predict
  int predictedLabel = gb.predict(sample);

  print('Predicted label: $predictedLabel');
}
```
Neural Networks:
```dart
import 'package:dartML/models/supervised/nn.dart';

void main() {
  // Load your dataset and preprocess it as needed

  List<List<double>> X = ...; // Features
  List<int> y = ...; // Labels

  // Create a NeuralNetwork instance
  NeuralNetwork nn = NeuralNetwork();

  // Define the network architecture
  nn.addLayer(10, activation: ActivationFunction.relu);
  nn.addLayer(1, activation: ActivationFunction.sigmoid);

  // Train the model
  nn.fit(X, y, numEpochs: 100, learningRate: 0.01);

  // Make predictions
  List<double> sample = ...; // Sample to predict
  double predictedValue = nn.predict(sample);

  print('Predicted value: $predictedValue');
}

Propensity Score Matching (PSM):
```dart
import 'package:dartML/models/supervised/psm.dart';

void main() {
// Load your dataset and preprocess it as needed

List<List<double>> X = ...; // Features
List<int> treatment = ...; // Treatment indicator

// Create a PropensityScoreMatching instance
PropensityScoreMatching psm = PropensityScoreMatching();

// Calculate propensity scores
psm.calculatePropensityScores(X, treatment);

// Perform matching
List<List<double>> matchedData = psm.performMatching(X, treatment, numMatches: 1);

print('Matched Data:');
for (List<double> sample in matchedData) {
print(sample);
}
}
```

### Evaluation Metrics
Accuracy:
```dart
import 'package:dartML/metrics/accuracy.dart';

void main() {
  List<int> trueLabels = ...; // True labels
  List<int> predictedLabels = ...; // Predicted labels

  // Calculate accuracy
  double accuracy = Accuracy.calculate(trueLabels, predictedLabels);

  print('Accuracy: $accuracy');
}
```

Confusion Matrix:
```dart
import 'package:dartML/metrics/confusion_matrix.dart';

void main() {
  List<int> trueLabels = ...; // True labels
  List<int> predictedLabels = ...; // Predicted labels

  // Calculate confusion matrix
  Map<String, dynamic> matrix = ConfusionMatrix.calculate(trueLabels, predictedLabels);

  // Print confusion matrix
  print('Confusion Matrix:');
  print(matrix);
}
```
Please note that the model in kFoldCrossValidation is of dynamic type. It is your responsibility to ensure that you pass objects that have fit and predict methods.

## Contributing
Contributions to dartML are welcome! If you find any issues or have suggestions for new features, please create an issue in the GitHub repository.

## License
This project is licensed under the MIT License.