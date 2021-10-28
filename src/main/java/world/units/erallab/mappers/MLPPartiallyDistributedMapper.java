package world.units.erallab.mappers;


import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import world.units.erallab.PartiallyDistributedSensing;

import java.util.List;
import java.util.Objects;


public class MLPPartiallyDistributedMapper extends AbstractPartiallyDistributedMapper<MultiLayerPerceptron> {

  private final String distribution;

  public MLPPartiallyDistributedMapper(Grid<? extends SensingVoxel> b, String config) {
    super(b, (config.contains("none")) ? 0 : 1, config.split("-")[0]);
    this.distribution = config.split("-")[1];
    if (!(this.distribution.equals("homo") || this.distribution.equals("hetero"))) {
      throw new IllegalArgumentException(String.format("Distribution model not known: %s", this.distribution));
    }
  }

  @Override
  public MultiLayerPerceptron getFunction(PartiallyDistributedSensing controller, Grid.Entry<? extends SensingVoxel> entry) {
    int inputs = controller.nOfInputs(entry.getX(), entry.getY());
    return new MultiLayerPerceptron(
            MultiLayerPerceptron.ActivationFunction.TANH,
            inputs,
            new int[]{},
            controller.nOfOutputs(entry.getX(), entry.getY())
    );
  }

  @Override
  public void setFuncParams(MultiLayerPerceptron function, List<Double> genotype, int num) {
    if (this.distribution.equals("homo")) {
      function.setParams(genotype.stream().mapToDouble(d -> d).toArray());
    }
    else {
      int genPerVoxels = this.getGenotypeSizeForVoxel();
      function.setParams(genotype.subList(num * genPerVoxels, (num + 1) * genPerVoxels).stream().mapToDouble(d -> d).toArray());
    }
  }

  public int getGenotypeSizeForVoxel() {
    int sum = 0;
    for (Grid.Entry<? extends SensingVoxel> entry : this.body) {
      if (entry.getValue() == null) {
        continue;
      }
      int inputs = PartiallyDistributedSensing.inputs(entry.getValue(), AbstractPartiallyDistributedMapper.getNumberNeighbors(this.neighborConfig, this.body));
      sum += MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(inputs, new int[]{}, /*getNumberNeighbors(config, body) + 1));*/(this.neighborConfig.contains("none") ? 1 : 2)));
      break;
    }
    return sum;
  }

  @Override
  public int getGenotypeSize() {
      int sum = this.getGenotypeSizeForVoxel();
      return (this.distribution.equals("hetero") ? sum * (int) body.count(Objects::nonNull) : sum);
  }

}
