package world.units.erallab;

import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;

// TODO: more sophisticated takes numAttention as input
public class ModuleGaussianMutation implements Mutation<List<Double>> {

  private final Mutation<List<Double>> innerMutation;
  private final int numAttention;

  public ModuleGaussianMutation(double sigma, int numAttention) {
    this.innerMutation = new GaussianMutation(sigma);
    this.numAttention = numAttention;
  }

  @Override
  public List<Double> mutate(List<Double> parent, Random random) {
    List<Double> newBorn = new ArrayList<>();
    for (int i = 0; i < this.numAttention; ++i) {
      newBorn.add(parent.get(i));
    }
    newBorn.addAll(this.innerMutation.mutate(parent.subList(this.numAttention, parent.size()), random));
    return newBorn;
  }

}
