/*
 * Copyright (C) 2020 Eric Medvet <eric.medvet@gmail.com> (as Eric Medvet <eric.medvet@gmail.com>)
 *
 * This program is free software: you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 * See the GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */
package world.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.*;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;
import org.apache.commons.lang3.ArrayUtils;
import org.apache.commons.math3.util.Pair;
import world.units.erallab.mappers.AbstractPartiallyDistributedMapper;

import java.util.Arrays;
import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;

/**
 * @author federico
 */
public class PartiallyDistributedSensing extends AbstractController<SensingVoxel> implements Snapshottable {

  private static class FunctionWrapper implements TimedRealFunction {
    @JsonProperty
    private final TimedRealFunction inner;

    @JsonCreator
    public FunctionWrapper(@JsonProperty("inner") TimedRealFunction inner) {
      this.inner = inner;
    }

    @Override
    public double[] apply(double t, double[] in) {
      return inner.apply(t, in);
    }

    @Override
    public int getInputDimension() {
      return inner.getInputDimension();
    }

    @Override
    public int getOutputDimension() {
      return inner.getOutputDimension();
    }
  }

  @JsonProperty
  private final int signals;
  @JsonProperty
  private final Grid<Integer> nOfInputGrid;
  @JsonProperty
  private final Grid<TimedRealFunction> functions;
  @JsonProperty
  private final String config;

  private final BiFunction<Pair<Integer, Integer>, Grid<? extends SensingVoxel>, List<Pair<Integer, Integer>>> neighborhood;

  private final Grid<double[]> lastSignalsGrid;

  private final Grid<double[]> currSignalsGrid;

  private final Grid<Double> outputGrid;
  private static final int nMessageBins = 25;
  private int[] nMessages;
  private double lastT = Double.NEGATIVE_INFINITY;
  private int downsamplingScale;
  private int originalVoxels;

  public static int inputs(SensingVoxel voxel, int nNeighbors) {
    return nNeighbors + voxel.getSensors().stream().mapToInt(s -> s.getDomains().length).sum();
  }

  @JsonCreator
  public PartiallyDistributedSensing(
          @JsonProperty("signals") int signals,
          @JsonProperty("nOfInputGrid") Grid<Integer> nOfInputGrid,
          @JsonProperty("functions") Grid<TimedRealFunction> functions,
          @JsonProperty("config") String config
  ) {
    this.signals = signals;
    this.nOfInputGrid = nOfInputGrid;
    this.functions = functions;
    this.lastSignalsGrid = Grid.create(functions, f -> new double[signals]);
    this.currSignalsGrid = Grid.create(functions, f -> new double[signals]);
    this.config = config;
    this.neighborhood = AbstractPartiallyDistributedMapper.getNeighborhood(config);
    this.outputGrid = Grid.create(functions, f -> 0.0);
    this.downsamplingScale = 1;
    this.reset();
  }

  public PartiallyDistributedSensing(Grid<? extends SensingVoxel> voxels, int signals, String config, int nNeighbors) {
    this(
            signals,
            Grid.create(voxels.getW(), voxels.getH(), (x, y) -> voxels.get(x, y) == null ? 0 : inputs(voxels.get(x, y), nNeighbors) * (int) voxels.count(Objects::nonNull)),
            Grid.create(
                    voxels.getW(),
                    voxels.getH(),
                    (x, y) -> voxels.get(x, y) == null ? null : new FunctionWrapper(RealFunction.build(
                            (double[] in) -> new double[1 + signals * nNeighbors],
                            inputs(voxels.get(x, y), nNeighbors) * (int) voxels.count(Objects::nonNull),
                            1 + signals * nNeighbors)
                    )
            ),
            config
    );
  }

  public void setDownsamplingParams(int downsamplingScale, int originalVoxels) {
    this.downsamplingScale = downsamplingScale;
    this.originalVoxels = originalVoxels;
  }

  public Grid<Integer> getInputGrid() {
    return this.nOfInputGrid;
  }

  public Grid<TimedRealFunction> getFunctions() {
    return this.functions;
  }

  public Grid<double[]> getLastSignalsGrid() {
    return this.lastSignalsGrid;
  }

  @Override
  public void reset() {
    for (int x = 0; x < this.lastSignalsGrid.getW(); x++) {
      for (int y = 0; y < this.lastSignalsGrid.getH(); y++) {
        this.lastSignalsGrid.set(x, y, new double[this.nOfOutputs(x, y) - 1]);
      }
    }
    for (int x = 0; x < this.currSignalsGrid.getW(); x++) {
      for (int y = 0; y < this.currSignalsGrid.getH(); y++) {
        this.currSignalsGrid.set(x, y, new double[this.nOfOutputs(x, y) - 1]);
      }
    }
    this.functions.values().stream().filter(Objects::nonNull).forEach(f -> {
      if (f instanceof Resettable) {
        ((Resettable) f).reset();
      }
    });
    for (int x = 0; x < this.outputGrid.getW(); x++) {
      for (int y = 0; y < this.outputGrid.getH(); y++) {
        this.outputGrid.set(x, y, 0.0);
      }
    }
    this.nMessages = new int[nMessageBins];
  }

  @Override
  public Grid<Double> computeControlSignals(double t, Grid<? extends SensingVoxel> voxels) {
    //else if (t - this.lastT < 0.33) {
    //  return this.outputGrid;
    //}
    int i = 0;
    int nVoxels = (int) voxels.count(Objects::nonNull);
    for (Grid.Entry<? extends SensingVoxel> entry : voxels) {
      if (entry.getValue() == null) {
        continue;
      }
      //get inputs
      double[] signals = this.getLastSignals(entry.getX(), entry.getY(), voxels);
      this.updateBins(signals);
      double[] inputs = entry.getValue().getSensorReadings();
      inputs = ArrayUtils.addAll(inputs, signals);
      //compute outputs
      TimedRealFunction function = this.functions.get(entry.getX(), entry.getY());
      if (function instanceof SelfAttention) {
        ((SelfAttention) function).setId(i);
      }
      double[] processedInputs = positionalEncoding(inputs, nVoxels, i++);//, nVoxels, this.originalVoxels, this.downsamplingScale);
      double[] outputs = function != null ? function.apply(t, processedInputs) : new double[this.nOfOutputs(entry.getX(), entry.getY())];
      //apply outputs
      this.outputGrid.set(entry.getX(), entry.getY(), outputs[0]);
      System.arraycopy(outputs, 1, this.currSignalsGrid.get(entry.getX(), entry.getY()), 0, this.nOfOutputs(entry.getX(), entry.getY()) - 1);
    }
    for (Grid.Entry<? extends SensingVoxel>  entry : voxels) {
      if (entry.getValue() == null) {
        continue;
      }
      int x = entry.getX();
      int y = entry.getY();
      System.arraycopy(this.currSignalsGrid.get(x, y), 0,  this.lastSignalsGrid.get(x, y), 0, this.nOfOutputs(x, y) - 1);
    }
    this.lastT = t;
    return this.outputGrid;
  }

  private void updateBins(double[] signals) {
    double step = 2.0 / nMessageBins;
    for (double signal : signals) {
      for (int i = 1; i <= nMessageBins; ++i) {
        if (signal <= (step * i) - 1.0) {
          this.nMessages[i - 1] += 1;
          break;
        }
      }
    }
  }

  public double getUniformity() {
    double l2 = 0.0;
    int sum = Arrays.stream(this.nMessages).sum();
    for (int bin : this.nMessages) {
      l2 += Math.pow((double) bin / sum, 2);
    }
    return ((Math.sqrt(l2) * Math.sqrt(nMessageBins)) - 1) / (Math.sqrt(nMessageBins) - 1);
  }

  public static double[] positionalEncoding(double[] inputs, int n, int i) {
    double[] pos = new double[n * inputs.length];
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < inputs.length; ++k) {
        pos[j * inputs.length + k] = (i == j) ? inputs[k] : 0.0;
      }
    }
    return pos;
  }

  public static double[] frequencyEncoding(double[] inputs, int n, int i) {
    /*double[] pos = new double[n];
    for (int j = 0; j < n; ++j) {
      pos[j] = Math.cos(i / Math.pow(10000, (2.0 * j) / inputs.length));
    }
    return SelfAttention.flat(SelfAttention.matrixMult(SelfAttention.reshapeVector(pos, pos.length, 1), SelfAttention.reshapeVector(inputs, 1, inputs.length)));*/
    double[][] pos = new double[n][inputs.length];
    for (int j = 0; j < n; ++j) {
      for (int k = 0; k < inputs.length; ++k) {
        if (i == j) {
          pos[j][k] = inputs[k];
        }
        else {
          double value = j / Math.pow(10000, (k * 2.0) / inputs.length);
          pos[j][k] = (k % 2 == 0) ? Math.sin(value) : Math.cos(value);
        }
      }
    }
    return SelfAttention.flat(SelfAttention.matrixTranspose(pos));
  }

  private static double[] downsample(double[] inputs, int n, int originalN, int scale) {
    int dim = inputs.length / n;
    double[] output = new double[originalN * dim];
    int k = 0;
    for (int i = 0; i < originalN; ++i) {
      for (int j = 0; j < dim; ++j) {
        for (int p = 0; p < scale && j + p < dim; ++p) {
          output[i * dim + j] += inputs[k * dim + j + p];
        }
        output[i * dim + j] /= scale;
      }
      k += scale;
    }
    return output;
  }

  public int nOfInputs(int x, int y) {
    return this.nOfInputGrid.get(x, y);
  }

  public int nOfOutputs(int x, int y) {
    return this.signals + 1;
  }

  private double[] getLastSignals(int x, int y, Grid<? extends SensingVoxel> voxels) {
    List<Pair<Integer, Integer>> neighbors = this.neighborhood.apply(new Pair<>(x, y), voxels);
    double[] values = new double[neighbors.size()];
    if (this.signals <= 0) {
      return values;
    }
    int c = 0;
    for (Pair<Integer, Integer> entry : neighbors) {
      int adjacentX = entry.getFirst();
      int adjacentY = entry.getSecond();
      double[] lastSignals;
      if (adjacentX < 0 || adjacentX >= voxels.getW() || adjacentY < 0 || adjacentY >= voxels.getH() || voxels.get(adjacentX, adjacentY) == null) {
        lastSignals = new double[this.nOfOutputs(x, y) - 1];
      }
      else {
        lastSignals = this.lastSignalsGrid.get(adjacentX, adjacentY);
      }
      System.arraycopy(lastSignals, 0, values, c, this.signals);
      c = c + this.signals;
    }
    return values;
  }

  @Override
  public String toString() {
    return "PartiallyDistributedSensing{" +
            "signals=" + this.signals +
            ", functions=" + this.functions +
            '}';
  }

  @Override
  public Snapshot getSnapshot() {
    return new Snapshot(new DistributedSensingShape(this.functions, this.lastSignalsGrid), this.getClass());
  }

}