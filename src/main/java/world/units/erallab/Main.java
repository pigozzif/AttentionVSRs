package world.units.erallab;

import com.google.common.base.Stopwatch;
import com.google.common.collect.Range;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.malelab.jgea.Worker;
import it.units.malelab.jgea.core.IndependentFactory;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.*;
import it.units.malelab.jgea.core.evolver.stopcondition.FitnessEvaluations;
import it.units.malelab.jgea.core.listener.CSVPrinter;
import it.units.malelab.jgea.core.listener.Listener;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.operator.GeneticOperator;
import it.units.malelab.jgea.core.order.PartialComparator;
import it.units.malelab.jgea.core.selector.Tournament;
import it.units.malelab.jgea.core.selector.Worst;
import it.units.malelab.jgea.core.util.Args;
import it.units.malelab.jgea.core.util.Misc;

import it.units.malelab.jgea.representation.sequence.FixedLengthListFactory;
import it.units.malelab.jgea.representation.sequence.numeric.GaussianMutation;
import it.units.malelab.jgea.representation.sequence.numeric.GeometricCrossover;
import it.units.malelab.jgea.representation.sequence.numeric.UniformDoubleFactory;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.function.Function;

import org.dyn4j.dynamics.Settings;
import world.units.erallab.mappers.AbstractPartiallyDistributedMapper;
import world.units.erallab.mappers.GenotypeSized;
import world.units.erallab.mappers.SelfAttentionPartiallyDistributedMapper;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;


public class Main extends Worker {

  private static int seed;
  private static String evolverName;
  private static int nEvals;
  private static double episodeTime;
  private static String terrain;
  private static String shape;
  private static String config;
  private static String exp;
  private static String sensorConfig;
  private static final double frequencyThreshold = 10.0D;
  private static final int nFrequencySamples = 100;
  private static String  bestFileName = "./output/";
  private static Settings physicsSettings;
  private static double stepSize;

  public Main(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new Main(args);
  }

  public void run() {
    seed = Args.i(this.a("seed", null));
    evolverName = this.a("evolver", null);
    shape = this.a("shape", null);
    config = this.a("config", null);
    exp = this.a("exp", null);
    stepSize = Double.parseDouble(this.a("step", "0.35"));
    terrain = this.a("terrain", "hilly-1-10-rnd");
    sensorConfig = this.a("sensors", "uniform-a+vxy+t+px+py-0.01");
    episodeTime = 30.0D;
    nEvals = 30000;
    physicsSettings = new Settings();
    bestFileName += String.join(".", "best", evolverName, String.valueOf(seed), exp, config, shape, "csv");

    try {
      this.evolve();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  private void evolve() throws FileNotFoundException {
    Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(shape));
    Function<List<Double>, Robot<?>> mapper = AbstractPartiallyDistributedMapper.mapperFactory(exp, body, config);
    IndependentFactory<List<Double>> factory = new FixedLengthListFactory<>(((GenotypeSized) mapper).getGenotypeSize(), new UniformDoubleFactory(-1.0D, 1.0D));
    Function<Robot<?>, Outcome> trainingTask = buildLocomotionTask(new Random(seed));

    try {
      Stopwatch stopwatch = Stopwatch.createStarted();
      L.info(String.format("Starting %s", bestFileName));
      Collection<Robot<?>> solutions = switch (evolverName) {
        case "cmaes" -> this.evolveCMAES(factory, mapper, trainingTask);
        case "es" -> this.evolveES(factory, mapper, trainingTask);
        case "ga" -> this.evolveGA(factory, mapper, trainingTask, Map.of(new GaussianMutation(0.35D), 0.2D, new GeometricCrossover(Range.closed(-0.5D, 1.5D)).andThen(new GaussianMutation(0.1D)), 0.8D));
        case "ga-mut" -> this.evolveGA(factory, mapper, trainingTask, Map.of(new GaussianMutation(stepSize), 1.0D));
        case "ga-mod-mut" -> this.evolveGA(factory, mapper, trainingTask, Map.of(new ModuleGaussianMutation(0.35D, ((SelfAttentionPartiallyDistributedMapper) mapper).getAttentionSizeForVoxel()), 1.0));
        case "ga-mix-cx" -> this.evolveGA(factory, mapper, trainingTask, Map.of(new ModuleCrossover(((SelfAttentionPartiallyDistributedMapper) mapper).getAttentionSizeForVoxel()).andThen(new GaussianMutation(0.1D)), 8.0D, new GaussianMutation(0.35D), 0.2D));
        default -> throw new IllegalStateException(String.format("Evolver not known: %s", evolverName));
      };
      L.info(String.format("Done %s: %d solutions in %4ds", bestFileName, solutions.size(), stopwatch.elapsed(TimeUnit.SECONDS)));
    }
    catch (ExecutionException | InterruptedException e) {
      L.severe(String.format("Cannot complete %s due to %s", bestFileName, e));
      e.printStackTrace();
    }
  }

  private Collection<Robot<?>> evolveCMAES(IndependentFactory<List<Double>> factory, Function<List<Double>, Robot<?>> mapper, Function<Robot<?>, Outcome> trainingTask) throws ExecutionException, InterruptedException {
    Evolver<List<Double>, Robot<?>, Outcome> evolver = new CMAESEvolver<>(mapper, factory, PartialComparator.from(Double.class).reversed().comparing(i -> i.getFitness().getVelocity()));
    return evolver.solve(trainingTask, new FitnessEvaluations(nEvals), new Random(seed), this.executorService, createListenerFactory().build());
  }

  private Collection<Robot<?>> evolveES(IndependentFactory<List<Double>> factory, Function<List<Double>, Robot<?>> mapper, Function<Robot<?>, Outcome> trainingTask) throws ExecutionException, InterruptedException {
    Evolver<List<Double>, Robot<?>, Outcome> evolver = new BasicEvolutionaryStrategy<>(mapper, factory, PartialComparator.from(Double.class).reversed().comparing(i -> i.getFitness().getVelocity()), 0.35D, 40, 10, 1, true);
    return evolver.solve(trainingTask, new FitnessEvaluations(nEvals), new Random(seed), this.executorService, createListenerFactory().build());
  }

  private Collection<Robot<?>> evolveGA(IndependentFactory<List<Double>> factory, Function<List<Double>, Robot<?>> mapper, Function<Robot<?>, Outcome> trainingTask, Map<GeneticOperator<List<Double>>, Double> operatorMap) throws ExecutionException, InterruptedException {
    Evolver<List<Double>, Robot<?>, Outcome> evolver = new StandardEvolver<>(mapper, factory, PartialComparator.from(Double.class).reversed().comparing(i -> i.getFitness().getVelocity()), 100, operatorMap, new Tournament(5), new Worst(), 100, true, !terrain.contains("flat"));
    return evolver.solve(trainingTask, new FitnessEvaluations(nEvals), new Random(seed), this.executorService, createListenerFactory().build());
  }

  private Listener.Factory<Event<?, ? extends Robot<?>, ? extends Outcome>> createListenerFactory() {
    Function<Outcome, Double> fitnessFunction = Outcome::getVelocity;
    // consumers
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions = AuxUtils.basicFunctions();
    List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions = AuxUtils.populationFunctions(fitnessFunction);
    List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> individualFunctions = AuxUtils.individualFunctions(fitnessFunction);
    List<NamedFunction<Outcome, ?>> basicOutcomeFunctions = AuxUtils.basicOutcomeFunctions();
    List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions = AuxUtils.detailedOutcomeFunctions(0, frequencyThreshold, nFrequencySamples);
    // file listener (one best per iteration)
    return new CSVPrinter<>(Misc.concat(List.of(
            basicFunctions,
            populationFunctions,
            NamedFunction.then(best(), individualFunctions),
            NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), basicOutcomeFunctions),
            NamedFunction.then(as(Outcome.class).of(fitness()).of(best()), detailedOutcomeFunctions),
            NamedFunction.then(best(), AuxUtils.serializationFunction(true))
    )), new File(bestFileName)
    );
  }

  public static Function<Robot<?>, Outcome> buildLocomotionTask(Random random) {
    return buildLocomotionTask(terrain, episodeTime, random, physicsSettings);
  }

  public static Function<Robot<?>, Outcome> buildLocomotionTask(String terrain, double episodeTime, Random random, Settings physicsSettings) {
    if (!terrain.contains("-rnd")) {
      return Misc.cached(new Locomotion(
              episodeTime,
              Locomotion.createTerrain(terrain),
              physicsSettings
      ), 0);
    }
    return r -> new Locomotion(
            episodeTime,
            Locomotion.createTerrain(terrain.replace("-rnd", "-" + random.nextInt(10000))),
            physicsSettings
    ).apply(r);
  }

}
