package world.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.snapshots.MLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Domain;
import it.units.erallab.hmsrobots.util.Parametrized;

import java.io.Serializable;
import java.util.Arrays;
import java.util.stream.DoubleStream;


public class RecurrentNeuralNetwork implements Serializable, Parametrized, TimedRealFunction, Snapshottable {

  @JsonProperty
  private final MultiLayerPerceptron inputGate;
  @JsonProperty
  private final MultiLayerPerceptron.ActivationFunction hiddenActivationFunction;
  @JsonProperty
  private final double[][] hiddenWeights;
  @JsonProperty
  private final MultiLayerPerceptron outputGate;
  @JsonProperty
  private final double[] hiddenState;
  @JsonProperty
  private final int steps;
  private final double[][] memory; // memory[0] is current obs
  private double lastT;
  private double[] lastOutput;

  @JsonCreator
  public RecurrentNeuralNetwork(@JsonProperty("inputGate") MultiLayerPerceptron inputGate,
                                @JsonProperty("hiddenActivationFunction")MultiLayerPerceptron.ActivationFunction hiddenActivationFunction,
                                @JsonProperty("hiddenWeights") double[][] hiddenWeights,
                                @JsonProperty("outputGate") MultiLayerPerceptron outputGate,
                                @JsonProperty("hiddenState") double[] hiddenState,
                                @JsonProperty("steps") int steps) {
    this.inputGate = inputGate;
    this.hiddenActivationFunction = hiddenActivationFunction;
    this.hiddenWeights = hiddenWeights;
    this.outputGate = outputGate;
    this.hiddenState = hiddenState;
    this.steps = steps;
    this.memory = new double[steps][getInputDimension()];
    this.lastT = 0.0;
    this.lastOutput = new double[getOutputDimension()];
  }

  public RecurrentNeuralNetwork(MultiLayerPerceptron.ActivationFunction hiddenActivationFunction, MultiLayerPerceptron.ActivationFunction outputActivationFunction,
                                int nOfInputs, int nOfHiddenUnits, int nOfOutputs, int steps) {
    this(new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.IDENTITY, nOfInputs, new int[]{}, nOfHiddenUnits), hiddenActivationFunction, new double[nOfHiddenUnits][nOfHiddenUnits], new MultiLayerPerceptron(outputActivationFunction, nOfHiddenUnits, new int[]{}, nOfOutputs),
            new double[nOfHiddenUnits], steps);
  }

  public static int countWeights(int nOfInputs, int nHiddenUnits, int nOfOutputs) {
    return (nOfInputs * nHiddenUnits) + nHiddenUnits + (nHiddenUnits * nHiddenUnits) + (nHiddenUnits * nOfOutputs) + nOfOutputs;
  }

  @Override
  public double[] apply(double t, double[] inputs) {
    if (t - lastT < 0.33) {
      return lastOutput;
    }
    lastT = t;
    reshapeMemory(inputs);
    Arrays.fill(hiddenState, 0.0);
    // unroll through time
    for (int s = steps - 1; s >= 0; --s) {
      double[] hiddenBias = inputGate.apply(memory[s]);
      for (int j = 0; j < hiddenWeights.length; j++) {
        double sum = hiddenBias[j]; // set the bias
        for (int k = 0; k < hiddenWeights[j].length; ++k) {
          sum = sum + hiddenState[j] * hiddenWeights[j][k];
        }
        hiddenState[j] = hiddenActivationFunction.apply(sum);
      }
    }
    lastOutput = outputGate.apply(hiddenState);
    return lastOutput;
  }

  public void reshapeMemory(double[] currInput) {
    int inputDimension = getInputDimension();
    for (int i = steps - 2; i >= 0; --i) {
      if (i == 0) {
        System.arraycopy(currInput, 0, memory[i], 0, inputDimension);
      }
      else {
        System.arraycopy(memory[i], 0, memory[i + 1], 0, inputDimension);
      }
    }
  }

  @Override
  public int getInputDimension() {
    return this.inputGate.getInputDimension();
  }

  @Override
  public int getOutputDimension() {
    return this.outputGate.getOutputDimension();
  }

  @Override
  public Snapshot getSnapshot() {
    double[][] innerActivations = new double[1][];
    innerActivations[0] = hiddenState;
    double[][][] weights = new double[3][][];
    weights[0] = inputGate.getWeights()[0];
    weights[1] = hiddenWeights;
    weights[2] = outputGate.getWeights()[2];
    return new Snapshot(new MLPState(innerActivations, weights, Domain.of(-1D, 1D)), getClass());
  }

  @Override
  public double[] getParams() {
    return DoubleStream.concat(Arrays.stream(inputGate.getParams()), DoubleStream.concat(Arrays.stream(hiddenWeights).flatMapToDouble(DoubleStream::of), Arrays.stream(outputGate.getParams()))).toArray();
  }

  @Override
  public void setParams(double[] params) {
    int numInputGateParams = inputGate.getParams().length;
    int numHiddenGateParams = hiddenWeights.length * hiddenWeights.length;
    inputGate.setParams(Arrays.stream(params).limit(numInputGateParams).toArray());
    int p = 0;
    for (double[] hiddenWeight : hiddenWeights) {
      System.arraycopy(params, numInputGateParams + p, hiddenWeight, 0, hiddenWeight.length);
      p = p + hiddenWeight.length;
    }
    outputGate.setParams(Arrays.stream(params).skip(numInputGateParams + numHiddenGateParams).toArray());
  }

}
