package world.units.erallab;

import com.google.common.collect.Range;
import it.units.malelab.jgea.core.operator.Crossover;
import it.units.malelab.jgea.core.operator.Mutation;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class ModuleCrossover implements Crossover<List<Double>> {

  private final Crossover<List<Double>> innerCrossover;
  private final Mutation<List<Double>> innerMutation;
  private final int numAttention;

  public ModuleCrossover(double lower, double upper, double sigma, int numAttention) {
    this.innerCrossover = new GeometricCrossover(Range.closed(lower, upper));
    this.innerMutation = new GaussianMutation(sigma);
    this.numAttention = numAttention;
  }

  @Override
  public List<Double> recombine(List<Double> parent1, List<Double> parent2, Random random) {
    List<Double> newBorn = new ArrayList<>();
    for (int i = 0; i < numAttention; ++i) {
      newBorn.add(parent1.get(i));
    }
    newBorn.addAll(this.innerMutation.mutate(this.innerCrossover.recombine(parent1.subList(this.numAttention, parent1.size()),
            parent2.subList(this.numAttention, parent2.size()), random), random));
    return newBorn;
  }

}
