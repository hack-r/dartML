class ReinforcementLearning {
  Map<String, double> qTable;
  double learningRate;
  double discountFactor;
  List<Map<String, dynamic>> transitions;

  ReinforcementLearning({this.learningRate = 0.1, this.discountFactor = 0.9}) {
    qTable = {};
    transitions = [];
  }

  void addTransition(String state, String action, double reward, String nextState) {
    transitions.add({'state': state, 'action': action, 'reward': reward, 'nextState': nextState});
  }

  void train(int numEpisodes) {
    for (int episode = 0; episode < numEpisodes; episode++) {
      String currentState = 'A';

      while (currentState != 'C') {
        String action = chooseAction(currentState);
        Map<String, dynamic> transition = getTransition(currentState, action);
        double reward = transition['reward'];
        String nextState = transition['nextState'];

        updateQValue(currentState, action, reward, nextState);

        currentState = nextState;
      }
    }
  }

  String chooseAction(String state) {
    // Exploration vs. Exploitation trade-off
    if (Random().nextDouble() < 0.1) {
      // Explore: Randomly select an action
      List<String> actions = getActions(state);
      return actions[Random().nextInt(actions.length)];
    } else {
      // Exploit: Select action with the highest Q-value
      double maxQValue = double.negativeInfinity;
      String bestAction = '';
      for (String action in getActions(state)) {
        double qValue = getQValue(state, action);
        if (qValue > maxQValue) {
          maxQValue = qValue;
          bestAction = action;
        }
      }
      return bestAction;
    }
  }

  List<String> getActions(String state) {
    List<String> actions = [];
    for (Map<String, dynamic> transition in transitions) {
      if (transition['state'] == state && !actions.contains(transition['action'])) {
        actions.add(transition['action']);
      }
    }
    return actions;
  }

  Map<String, dynamic> getTransition(String state, String action) {
    for (Map<String, dynamic> transition in transitions) {
      if (transition['state'] == state && transition['action'] == action) {
        return transition;
      }
    }
    return {};
  }

  double getQValue(String state, String action) {
    String qKey = '$state-$action';
    if (qTable.containsKey(qKey)) {
      return qTable[qKey]!;
    }
    return 0.0;
  }

  void updateQValue(String state, String action, double reward, String nextState) {
    double maxNextQValue = double.negativeInfinity;
    for (String nextAction in getActions(nextState)) {
      double nextQValue = getQValue(nextState, nextAction);
      if (nextQValue > maxNextQValue) {
        maxNextQValue = nextQValue;
      }
    }

    String qKey = '$state-$action';
    double oldQValue = getQValue(state, action);
    double newQValue = (1 - learningRate) * oldQValue + learningRate * (reward + discountFactor * maxNextQValue);
    qTable[qKey] = newQValue;
  }
}
