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
    int start, end;
    if (random.nextDouble() < 0.0) {
      start = 0;
      end = this.numAttention;
      newBorn = this.innerMutation.mutate(parent.subList(start, end), random);
      for (int i = end; i < parent.size(); ++i) {
        newBorn.add(parent.get(i));
      }
    }
    else {
      start = this.numAttention;
      end = parent.size();
      for (int i = 0; i < start; ++i) {
        newBorn.add(parent.get(i));
      }
      newBorn.addAll(this.innerMutation.mutate(parent.subList(start, end), random));
    }
    return newBorn;
  }

}
