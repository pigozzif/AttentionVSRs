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
  private static double episodeTime;
  private static String terrain;
  private static String shape;
  private static String config;
  private static String exp;
  private static String sensorConfig;
  private static int nEvals;
  private static boolean isFineTuning;
  private static String transformation;
  private static final double frequencyThreshold = 10.0D;
  private static final int nFrequencySamples = 100;
  private static String  bestFileName = "./output/";
  private static Settings physicsSettings;

  public Main(String[] args) {
    super(args);
  }

  public static void main(String[] args) {
    new Main(args);
  }

  public void run() {
    seed = Args.i(this.a("seed", null));
    shape = this.a("shape", null);
    config = this.a("config", null);
    exp = this.a("exp", null);
    terrain = this.a("terrain", "hilly-1-10-rnd");
    sensorConfig = this.a("sensors", "uniform-a+vxy+t-0.01");
    nEvals = Args.i(this.a("nevals", "30000"));
    isFineTuning = Boolean.parseBoolean(this.a("finetune", "false"));
    transformation = this.a("transformation", "identity");
    episodeTime = 30.0D;
    physicsSettings = new Settings();
    bestFileName += String.join(".", (isFineTuning) ? "finetune" : "best", String.valueOf(seed), exp, config, shape, sensorConfig.split("-")[0], "csv");

    try {
      this.evolve();
    } catch (FileNotFoundException e) {
      e.printStackTrace();
    }
  }

  private void evolve() throws FileNotFoundException {
    Grid<? extends SensingVoxel> body = AbstractPartiallyDistributedMapper.buildSensingGrid(sensorConfig).apply(RobotUtils.buildShape(shape));
    Function<List<Double>, Robot<?>> mapper = AbstractPartiallyDistributedMapper.mapperFactory(exp, body, config);
    IndependentFactory<List<Double>> factory = (!isFineTuning) ? new FixedLengthListFactory<>(((GenotypeSized) mapper).getGenotypeSize(), new UniformDoubleFactory(-1.0D, 1.0D)) :
            new ModuleIndependentFactory(new FixedLengthListFactory<>(((SelfAttentionPartiallyDistributedMapper) mapper).getValuesAndDownstreamSizeForVoxel(), new UniformDoubleFactory(-1.0D, 1.0D)), getAttentionToFineTune(bestFileName, shape, seed), body);
    Function<Robot<?>, Outcome> trainingTask = buildLocomotionTask(transformation, new Random(seed));

    try {
      Stopwatch stopwatch = Stopwatch.createStarted();
      L.info(String.format("Starting %s", bestFileName));
      Collection<Robot<?>> solutions = this.evolveGA(factory, mapper, trainingTask, (!isFineTuning) ? Map.of(new GaussianMutation(0.35D), 0.2D, new GeometricCrossover(Range.closed(-0.5D, 1.5D)).andThen(new GaussianMutation(0.1D)), 0.8D) : Map.of(new ModuleGaussianMutation(0.35D, ((SelfAttentionPartiallyDistributedMapper) mapper).getAttentionSizeForVoxel()), 0.2D, new ModuleCrossover(-0.5D, 1.5D, 0.1D, ((SelfAttentionPartiallyDistributedMapper) mapper).getAttentionSizeForVoxel()), 0.8D));
      L.info(String.format("Done %s: %d solutions in %4ds", bestFileName, solutions.size(), stopwatch.elapsed(TimeUnit.SECONDS)));
    }
    catch (ExecutionException | InterruptedException e) {
      L.severe(String.format("Cannot complete %s due to %s", bestFileName, e));
      e.printStackTrace();
    }
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

  public static Function<Robot<?>, Outcome> buildLocomotionTask(String transformation, Random random) {
    return buildLocomotionTask(terrain, episodeTime, random, physicsSettings).compose(RobotUtils.buildRobotTransformation(transformation, random));
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

  public static Robot<?> getAttentionToFineTune(String path, String shape, int seed) {
    String newPath = path.replace("finetune", "best");
    return SurrogateValidator.parseIndividualFromFile(newPath.replace(shape, VideoMaker.getOriginalShape(shape)), new Random(seed), 100);
  }

}
