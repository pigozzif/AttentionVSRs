package world.units.erallab.mappers;

import it.units.erallab.hmsrobots.core.controllers.*;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Parametrized;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.apache.commons.lang3.NotImplementedException;
import world.units.erallab.RecurrentNeuralNetwork;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;


public class RNNMapper implements Function<List<Double>, Robot<?>>, GenotypeSized {

  private final Grid<? extends SensingVoxel> body;
  private final int signals;
  private final String controllerType;
  private final int nHiddenUnits;
  private final int windowSize;
  private final double t;
  private final int nVoxels;
  private final int nSensors;

  public RNNMapper(Grid<? extends SensingVoxel> b, String config) {
    this.body = b;
    this.signals = config.contains("none") ? 0 : 1;
    this.controllerType = config.split("-")[0];
    this.nHiddenUnits = Integer.parseInt(config.split("-")[1]);
    this.windowSize = Integer.parseInt(config.split("-")[2]);
    this.t = 0.33;
    this.nVoxels = (int) this.body.count(Objects::nonNull);
    this.nSensors = this.body.get(0, 0).getSensors().stream().mapToInt(s -> s.getDomains().length).sum();
  }

  @Override
  public Robot<?> apply(List<Double> genotype) {
    if (genotype.size() != this.getGenotypeSize()) {
      throw new IllegalArgumentException(String.format("Wrong genotype size %d instead of %d", genotype.size(), this.getGenotypeSize()));
    }
    AbstractController controller;
    RecurrentNeuralNetwork function;
    switch (this.controllerType) {
      case "partially" -> throw new NotImplementedException("Partially distributed not yet implemented for RNN");
      case "centralized" -> {
        function = this.getFunction(this.nVoxels * this.nSensors, this.nVoxels);
        ((Parametrized) function).setParams(genotype.stream().mapToDouble(d -> d).toArray());
        controller = new CentralizedSensing(this.nVoxels * this.nSensors, this.nVoxels, function);
      }
      case "distributed" -> {
        controller = new DistributedSensing(this.body, this.signals);
        for (Grid.Entry<? extends SensingVoxel> entry : this.body) {
          if (entry.getValue() == null) {
            continue;
          }
          function = this.getFunction(this.nSensors, 5);
          ((Parametrized) function).setParams(genotype.stream().mapToDouble(d -> d).toArray());
          ((DistributedSensing) controller).getFunctions().set(entry.getX(), entry.getY(), function);
        }
      }
      default -> throw new IllegalArgumentException(String.format("Controller type for RNN not known: %s", this.controllerType));
    }
    return new Robot<>(new StepController<>(controller, this.t), SerializationUtils.clone(this.body));
  }

  public RecurrentNeuralNetwork getFunction(int nOfInputs, int nOfOutputs) {
    return new RecurrentNeuralNetwork(
            MultiLayerPerceptron.ActivationFunction.TANH,
            MultiLayerPerceptron.ActivationFunction.TANH,
            nOfInputs,
            this.nHiddenUnits,
            nOfOutputs,
            this.windowSize);
  }

  @Override
  public int getGenotypeSize() {
    int nOfInputs = (this.controllerType.equals("centralized")) ? this.nVoxels * this.nSensors : this.nSensors + 4;
    int nOfOutputs = (this.controllerType.equals("centralized")) ? this.nVoxels : 5;
    return RecurrentNeuralNetwork.countWeights(nOfInputs, this.nHiddenUnits, nOfOutputs);
  }

}
