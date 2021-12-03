package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.malelab.jgea.core.IndependentFactory;

import java.util.ArrayList;
import java.util.List;
import java.util.Objects;
import java.util.Random;


public class ModuleIndependentFactory implements IndependentFactory<List<Double>> {

  private final IndependentFactory<List<Double>> innerFactory;
  private final Robot<?> prototype;
  private final Grid<? extends SensingVoxel>  body;

  public ModuleIndependentFactory(IndependentFactory<List<Double>> innerFactory, Robot<?> prototype, Grid<? extends SensingVoxel> body) {
    this.innerFactory = innerFactory;
    this.prototype = prototype;
    this.body = body;
  }

  @Override
  public List<Double> build(Random random) {
    List<Double> newBorn = new ArrayList<>();
    List<Double> newGenes = this.innerFactory.build(random);
    SelfAttention attention = ((SelfAttention) ((PartiallyDistributedSensing) ((StepController) prototype.getController()).getInnerController()).getFunctions().get(0, 0));
    int k = 0;
    int nVoxelsBody = (int) body.count(Objects::nonNull);
    for (int i = 0; i < 16; ++i) {
      newBorn.add(attention.getQueriesAndKeysMatrices()[i]);
    }
    for (int i = 0; i < nVoxelsBody * 8; ++i) {
      newBorn.add(newGenes.get(k++));
    }
    for (int i = 0; i < 16; ++i) {
      newBorn.add(attention.getQueriesAndKeysBias()[i]);
    }
    newBorn.addAll(newGenes.subList(k, newGenes.size()));
    return newBorn;
  }

}
