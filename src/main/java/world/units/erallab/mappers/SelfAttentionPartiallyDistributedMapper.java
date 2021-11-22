package world.units.erallab.mappers;

import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import world.units.erallab.PartiallyDistributedSensing;
import world.units.erallab.SelfAttention;

import java.util.List;
import java.util.Objects;


public class SelfAttentionPartiallyDistributedMapper extends AbstractPartiallyDistributedMapper<SelfAttention> {

  private final int din;
  private final int dk;
  private final int dv;
  private final String distribution;
  private final boolean isTanh;
  private final boolean isNopos;

  public SelfAttentionPartiallyDistributedMapper(Grid<? extends SensingVoxel> b, String config) {
    super(b, config.contains("none") ? 0 : 1, config.split("-")[0]);
    int[] params = parseAttentionParams(config);
    this.din = params[0];
    this.dk = params[1];
    this.dv = params[2];
    this.distribution = config.split("-")[4];
    this.isTanh = config.contains("tanh");
    this.isNopos = config.contains("nopos");
    if (!(this.distribution.equals("homo|homo") || this.distribution.equals("hetero|homo") || this.distribution.equals("homo|hetero") || this.distribution.equals("hetero|hetero"))) {
      throw new IllegalArgumentException(String.format("Distribution model not known: %s", this.distribution));
    }
  }

  public static int[] parseAttentionParams(String config) {
    String[] params = config.split("-");
    return new int[] {Integer.parseInt(params[1]), Integer.parseInt(params[2]), Integer.parseInt(params[3])};
  }

  @Override
  public SelfAttention getFunction(PartiallyDistributedSensing controller, Grid.Entry<? extends SensingVoxel> entry) {
    int nVoxels = (int) this.body.count(Objects::nonNull);
    int mlpInput;
    if (this.isNopos) {
      mlpInput = this.din;
    }
    else if (this.isTanh) {
      mlpInput = nVoxels * this.din;
    }
    else {
      mlpInput = this.din * this.dv;
    }
    return new SelfAttention(new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, mlpInput, new int[]{}, controller.nOfOutputs(entry.getX(), entry.getY())),
              nVoxels, this.din, this.dk, this.dv);
  }

  @Override
  public void setFuncParams(SelfAttention function, List<Double> genotype, int num) {
    int numAttention = function.countParams();
    int numDownstream = this.getDownstreamSizeForVoxel();
    switch (this.distribution) {
      case "homo|homo" -> {
        function.setAttentionParams(genotype.subList(0, numAttention).stream().mapToDouble(d -> d).toArray());
        function.setDownstreamParams(genotype.subList(numAttention, genotype.size()).stream().mapToDouble(d -> d).toArray());
      }
      case "homo|hetero" -> {
        function.setAttentionParams(genotype.subList(0, numAttention).stream().mapToDouble(d -> d).toArray());
        function.setDownstreamParams(genotype.subList(numAttention + (num * numDownstream), numAttention + ((num + 1) * numDownstream)).stream().mapToDouble(d -> d).toArray());
      }
      case "hetero|homo" -> {
        function.setDownstreamParams(genotype.subList(0, numDownstream).stream().mapToDouble(d -> d).toArray());
        function.setAttentionParams(genotype.subList(numDownstream + (num * numAttention), numDownstream + ((num + 1) * numAttention)).stream().mapToDouble(d -> d).toArray());
      }
      default -> {
        List<Double> currGen = genotype.subList(num * this.getGenotypeSizeForVoxel(), (num + 1) * this.getGenotypeSizeForVoxel());
        function.setAttentionParams(currGen.subList(0, numAttention).stream().mapToDouble(d -> d).toArray());
        function.setDownstreamParams(currGen.subList(numAttention, currGen.size()).stream().mapToDouble(d -> d).toArray());
      }
    }
  }

  public int getGenotypeSizeForVoxel() {
    int sumAttention = this.getAttentionSizeForVoxel();
    int sumDownstream = this.getDownstreamSizeForVoxel();
    return sumAttention + sumDownstream;
  }

  public int getAttentionSizeForVoxel() {
    return SelfAttention.countParams(this.din, this.dk, this.dv, (int) this.body.count(Objects::nonNull));
  }

  public int getDownstreamSizeForVoxel() {
    int sumDownstream = 0;
    int nVoxels = (int) this.body.count(Objects::nonNull);
    for (Grid.Entry<? extends SensingVoxel> entry : this.body) {
      if (entry.getValue() == null) {
        continue;
      }
      int inputs;
      if (this.isNopos) {
        inputs = this.din;
      }
      else if (this.isTanh) {
        inputs = nVoxels * this.din;
      }
      else {
        inputs = this.din * this.dv;
      }
      sumDownstream += MultiLayerPerceptron.countWeights(MultiLayerPerceptron.countNeurons(inputs, new int[]{}, this.neighborConfig.contains("none") ? 1 : 2));
      break;
    }
    return sumDownstream;
  }

  @Override
  public int getGenotypeSize() {
    int sumAttention = this.getAttentionSizeForVoxel();
    int sumDownstream = this.getDownstreamSizeForVoxel();
    if (this.distribution.startsWith("hetero|")) {
      sumAttention = sumAttention * (int) body.count(Objects::nonNull);
    }
    if (this.distribution.endsWith("|hetero")) {
      sumDownstream = sumDownstream * (int) body.count(Objects::nonNull);
    }
    return sumAttention + sumDownstream;
  }

}
