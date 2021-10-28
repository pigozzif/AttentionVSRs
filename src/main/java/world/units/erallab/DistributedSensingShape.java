package world.units.erallab;


import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.geometry.Point2;
import it.units.erallab.hmsrobots.core.geometry.Shape;
import it.units.erallab.hmsrobots.util.Grid;


public class DistributedSensingShape implements Shape {

  private final Grid<TimedRealFunction> functionGrid;
  private final Grid<double[]> lastSignalsGrid;

  public DistributedSensingShape(Grid<TimedRealFunction> functionGrid, Grid<double[]> lastSignalsGrid) {
    this.functionGrid = functionGrid;
    this.lastSignalsGrid = lastSignalsGrid;
  }

  public Grid<TimedRealFunction> getFunctionGrid() { return this.functionGrid; }

  @Override
  public BoundingBox boundingBox() {
    return null;
  }

  @Override
  public Point2 center() {
    return null;
  }

}
