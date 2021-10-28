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

    public static int inputs(SensingVoxel voxel, int nNeighbors) {
        return (nNeighbors + voxel.getSensors().stream().mapToInt(s -> s.getDomains().length).sum()) * nNeighbors;
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
        this.reset();
    }

    public PartiallyDistributedSensing(Grid<? extends SensingVoxel> voxels, int signals, String config, int nNeighbors) {
        this(
                signals,
                Grid.create(voxels.getW(), voxels.getH(), (x, y) -> voxels.get(x, y) == null ? 0 : inputs(voxels.get(x, y), nNeighbors)),
                Grid.create(
                        voxels.getW(),
                        voxels.getH(),
                        (x, y) -> voxels.get(x, y) == null ? null : new FunctionWrapper(RealFunction.build(
                                (double[] in) -> new double[1 + signals * nNeighbors],
                                inputs(voxels.get(x, y), nNeighbors),
                                1 + signals * nNeighbors)
                        )
                ),
                config
        );
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
    }

    @Override
    public Grid<Double> computeControlSignals(double t, Grid<? extends SensingVoxel> voxels) {
        Grid<Double> outputGrid = Grid.create(voxels.getW(), voxels.getH(), (x, y) -> 0.0);
        int i = 0;
        for (Grid.Entry<? extends SensingVoxel> entry : voxels) {
            if (entry.getValue() == null) {
                continue;
            }
            //get inputs
            double[] signals = this.getLastSignals(entry.getX(), entry.getY(), voxels);
            double[] inputs = ArrayUtils.addAll(entry.getValue().getSensorReadings(), signals);
            //compute outputs
            TimedRealFunction function = this.functions.get(entry.getX(), entry.getY());
            double[] outputs = function != null ? function.apply(t, positionalEncoding(inputs, (int) voxels.count(Objects::nonNull), i++)) : new double[this.nOfOutputs(entry.getX(), entry.getY())];
            //apply outputs
            //entry.getValue().applyForce(outputs[0]);
            outputGrid.set(entry.getX(), entry.getY(), outputs[0]);
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
        return outputGrid;
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