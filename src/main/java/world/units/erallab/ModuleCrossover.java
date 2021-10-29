package world.units.erallab;

import it.units.malelab.jgea.core.operator.Crossover;

import java.util.ArrayList;
import java.util.List;
import java.util.Random;


public class ModuleCrossover implements Crossover<List<Double>> {

  private final int numAttention;

  public ModuleCrossover(int numAttention) {
    this.numAttention = numAttention;
  }

  @Override
  public List<Double> recombine(List<Double> parent1, List<Double> parent2, Random random) {
    List<Double> newBorn = new ArrayList<>();
    int start1, end1, start2, end2;
    if (random.nextDouble() < 0.5) {
      start1 = 0;
      end1 = this.numAttention;
      start2 = this.numAttention;
      end2 = parent2.size();
    }
    else {
      start1 = this.numAttention;
      end1 = parent1.size();
      start2 = 0;
      end2 = this.numAttention;
    }
    for (int i = start1; i < end1; ++i) {
      newBorn.add(parent1.get(i));
    }
    for (int j = start2; j < end2; ++j) {
      newBorn.add(parent2.get(j));
    }
    return newBorn;
  }

}
