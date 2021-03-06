package world.units.erallab.mappers;

import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.sensors.Constant;
import it.units.erallab.hmsrobots.core.sensors.Noisy;
import it.units.erallab.hmsrobots.core.sensors.Sensor;
import it.units.erallab.hmsrobots.util.*;
import org.apache.commons.math3.util.Pair;
import world.units.erallab.PartiallyDistributedSensing;

import java.util.List;
import java.util.Objects;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.erallab.hmsrobots.util.RobotUtils.sensor;

// TODO: assumes same sensor config and architecture for every voxel
public abstract class AbstractPartiallyDistributedMapper<T extends RealFunction & Parametrized> implements Function<List<Double>, Robot<?>>, GenotypeSized {

  protected final Grid<? extends SensingVoxel> body;
  protected final int signals;
  protected final int nNeighbors;
  protected final String neighborConfig;
  protected final double t;

  public AbstractPartiallyDistributedMapper(Grid<? extends SensingVoxel> b, int s, String neighborConfig) {
    this.body = b;
    this.signals = s;
    this.nNeighbors = getNumberNeighbors(neighborConfig, b);
    this.neighborConfig = neighborConfig;
    this.t = 0.33;
  }

  @Override
  public Robot<?> apply(List<Double> genotype) {
    if (genotype.size() != this.getGenotypeSize()) {
      throw new IllegalArgumentException(String.format("Wrong genotype size %d instead of %d", genotype.size(), this.getGenotypeSize()));
    }
    PartiallyDistributedSensing controller = new PartiallyDistributedSensing(this.body, this.signals, this.neighborConfig, this.nNeighbors);
    int num = 0;
    for (Grid.Entry<? extends SensingVoxel> entry : this.body) {
      if (entry.getValue() == null) {
        continue;
      }
      T function = this.getFunction(controller, entry);
      this.setFuncParams(function, genotype, num++);
      controller.getFunctions().set(entry.getX(), entry.getY(), function);
    }
    return new Robot<>(new StepController<>(controller, this.t), SerializationUtils.clone(this.body));
  }

  public abstract T getFunction(PartiallyDistributedSensing controller, Grid.Entry<? extends SensingVoxel> entry);

  public abstract void setFuncParams(T function, List<Double> genotype, int num);

  public static BiFunction<Pair<Integer, Integer>, Grid<? extends SensingVoxel>, List<Pair<Integer, Integer>>> getNeighborhood(String config) {
    return switch (config) {
      case "none" -> (x, y) -> List.of();
      case "neumann" -> (x, y) -> List.of(new Pair<>(x.getFirst(), x.getSecond() + 1), new Pair<>(x.getFirst() + 1, x.getSecond()), new Pair<>(x.getFirst(), x.getSecond() - 1), new Pair<>(x.getFirst() - 1, x.getSecond()));
      case "moore" -> (x, y) -> List.of(new Pair<>(x.getFirst() - 1, x.getSecond() + 1), new Pair<>(x.getFirst(), x.getSecond() + 1), new Pair<>(x.getFirst() + 1, x.getSecond() + 1), new Pair<>(x.getFirst() + 1, x.getSecond()), new Pair<>(x.getFirst() + 1, x.getSecond() - 1), new Pair<>(x.getFirst(), x.getSecond() - 1), new Pair<>(x.getFirst() - 1, x.getSecond() - 1), new Pair<>(x.getFirst() - 1, x.getSecond()));
      case "all" -> (x, y) -> y.stream().filter(v -> v.getValue() != null).map(e -> new Pair<>(e.getX(), e.getY()))/*.sorted(Comparator.comparing((Pair<Integer, Integer> z) -> Math.sqrt(Math.pow(z.getFirst() - x.getFirst(), 2) + Math.pow(z.getSecond() - x.getSecond(), 2))).thenComparing(Pair::getFirst).thenComparing(Pair::getSecond))*/.collect(Collectors.toList());
      default -> throw new RuntimeException(String.format("Configuration not known: %s", config));
    };
  }

  public static int getNumberNeighbors(String config, Grid<? extends SensingVoxel> body) {
    return switch (config) {
      case "none" -> 0;
      case "neumann" -> 4;
      case "moore" -> 8;
      case "all" -> (int) body.count(Objects::nonNull);
      default -> throw new RuntimeException(String.format("Configuration not known: %s", config));
    };
  }

  public static Function<List<Double>, Robot<?>> mapperFactory(String exp, Grid<? extends SensingVoxel> b, String config) {
    return switch (exp) {
      case "baseline" -> new MLPPartiallyDistributedMapper(b, config);
      case "attention" -> new SelfAttentionPartiallyDistributedMapper(b, config);
      case "centralized" -> new CentralizedMapper(b, config);
      case "rnn" -> new RNNMapper(b, config);
      default -> throw new RuntimeException(String.format("Unknown mapper: %s", exp));
    };
  }

  public static Function<Grid<Boolean>, Grid<? extends SensingVoxel>> buildSensingGrid(String config) {
    double noiseSigma = 0.01d;
    return switch (config) {
      case "high" -> body -> Grid.create(body.getW(), body.getH(),
              (x, y) -> {
                if (!body.get(x, y)) {
                  return null;
                }
                return new SensingVoxel(
                        Utils.ofNonNull(
                                sensorWrapper("a", x, y, body, true),
                                sensorWrapper("t", x, y, body, y == 0),
                                sensorWrapper("vxy", x, y, body, y == body.getH() - 1),
                                sensorWrapper("l5", x, y, body, x == body.getW() - 1)
                                ).stream()
                                .map(s -> (!(s instanceof Constant)) ? new Noisy(s, noiseSigma, 0) : s)
                                .collect(Collectors.toList())
                );
              }
      );
      case "medium" -> body -> Grid.create(body.getW(), body.getH(),
              (x, y) -> {
                if (!body.get(x, y)) {
                  return null;
                }
                return new SensingVoxel(
                        Utils.ofNonNull(
                                sensorWrapper("a", x, y, body, true),
                                sensorWrapper("t", x, y, body, y == 0),
                                sensorWrapper("vxy", x, y, body, y == body.getH() - 1)
                                ).stream()
                                .map(s -> (!(s instanceof Constant)) ? new Noisy(s, noiseSigma, 0) : s)
                                .collect(Collectors.toList())
                );
              }
      );
      case "low" -> body -> Grid.create(body.getW(), body.getH(),
              (x, y) -> {
                if (!body.get(x, y)) {
                  return null;
                }
                return new SensingVoxel(
                        Utils.ofNonNull(
                                sensorWrapper("a", x, y, body, y >= body.getH() / 2)
                                ).stream()
                                .map(s -> (!(s instanceof Constant)) ? new Noisy(s, noiseSigma, 0) : s)
                                .collect(Collectors.toList())
                );
              }
      );
      default -> RobotUtils.buildSensorizingFunction(config);
    };
  }

  public static Sensor sensorWrapper(String name, int x, int y, Grid<Boolean> body, boolean condition) {
    Sensor sensor = sensor(name, x, y, body);
    if (!condition) {
      return new Constant(IntStream.range(0, sensor.getDomains().length).mapToDouble(i -> 0.0).toArray());
    }
    return sensor;
  }

}
