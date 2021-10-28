package world.units.erallab.mappers;

import it.units.erallab.hmsrobots.core.controllers.CentralizedSensing;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Parametrized;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import world.units.erallab.SelfAttention;

import java.util.List;
import java.util.Objects;
import java.util.function.Function;


public class CentralizedMapper implements Function<List<Double>, Robot<?>>, GenotypeSized {

  private final Grid<? extends SensingVoxel> body;
  private final int din;
  private final int dk;
  private final int dv;
  private final int nVoxels;

  public CentralizedMapper(Grid<? extends SensingVoxel> b, String config) {
    this.body = b;
    int[] params = SelfAttentionPartiallyDistributedMapper.parseAttentionParams(config);
    this.din = params[0];
    this.dk = params[1];
    this.dv = params[2];
    this.nVoxels = (int) b.count(Objects::nonNull);
  }

  public Robot<?> apply(List<Double> genotype) {
    if (genotype.size() != this.getGenotypeSize()) {
      throw new IllegalArgumentException(String.format("Wrong genotype size %d instead of %d", genotype.size(), this.getGenotypeSize()));
    }
    CentralizedSensing controller = new CentralizedSensing(this.nVoxels * this.din, this.nVoxels, new SelfAttention(new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, nVoxels * this.dv, new int[]{}, this.nVoxels),
            this.nVoxels, this.din, this.dk, this.dv));
    ((Parametrized) controller.getFunction()).setParams(genotype.stream().mapToDouble(d -> d).toArray());
    return new Robot<>(new StepController<>(controller, 0.33), SerializationUtils.clone(body));
  }

  @Override
  public int getGenotypeSize() {
    int sumDownstream = 0;
    for (Grid.Entry<? extends SensingVoxel> entry : this.body) {
      if (entry.getValue() == null) {
        continue;
      }
      int inputs = this.nVoxels * this.dv;
      sumDownstream += MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(inputs, new int[]{}, this.nVoxels));
      break;
    }
    return sumDownstream + SelfAttention.countParams(this.din, this.dk, this.dv);
  }

}
