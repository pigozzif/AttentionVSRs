package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.*;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.util.Args;
import org.dyn4j.dynamics.Settings;
import world.units.erallab.mappers.AbstractPartiallyDistributedMapper;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Objects;
import java.util.stream.Collectors;


public class DifferentMorphologies {

  private static final String dir = "/Users/federicopigozzi/Desktop/geom+pos/";
  public static final String sensorConfig = "uniform-a+vxy+t+px+py-0.01";

  public static void main(String[] args) throws IOException {
    int scale = Integer.parseInt(Args.a(args, "scale", null));
    BufferedWriter writer = new BufferedWriter(new FileWriter("transfer.csv", false));
    writer.write("velocity;scale;config;shape;run;serialized\n");
    for (File file : Files.walk(Paths.get(dir)).filter(p -> Files.isRegularFile(p) && p.toString().contains("best") && p.toString().contains(".ga.")).map(Path::toFile).collect(Collectors.toList())) {
      String path = file.getPath();
      System.out.println(path);
      String seed = path.split("\\.")[2];
      String shape = path.split("\\.")[5];
      String config = path.split("\\.")[4];
      Robot<?> originalRobot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      Robot<?> newRobot = getNewRobot(originalRobot, shape, config, scale);
      Outcome data = getOutcomeFromSimulation(newRobot, seed);
      writer.write(String.join(";", String.valueOf(data.getVelocity()), String.valueOf(scale), config, shape, seed,
              SerializationUtils.serialize(newRobot, SerializationUtils.Mode.GZIPPED_JSON)) + "\n");
    }
    writer.close();
  }

  private static Outcome getOutcomeFromSimulation(Robot<?> robot, String seed) {
    Task<Robot<?>, Outcome> task = new Locomotion(30.0, Locomotion.createTerrain("hilly-1-10-" + seed), new Settings());
    return task.apply(robot);
  }

  private static String getSerializedFromFile(File file) throws IOException {
    BufferedReader reader = new BufferedReader(new FileReader(file));
    String lastLine = reader.lines().reduce((first, second) -> second).orElse(null);
    String serialized;
    if (lastLine != null) {
      String[] fragments = lastLine.split(";");
      serialized = fragments[fragments.length - 1];
    }
    else {
      throw new IllegalArgumentException("File  + " + file.getPath() + " is empty!");
    }
    reader.close();
    return serialized;
  }

  private static Robot<?> getNewRobot(Robot<?> originalRobot, String originalShape, String config, int scale) {
    String newShape = originalShape.split("-")[0];
    if (originalShape.equals("biped-4x3") && scale == 2) {
      newShape = /*"comb-7x2";*/newShape + "-6x4";
    }
    else if (originalShape.equals("comb-7x2") && scale == 2) {
      newShape = /*"biped-4x3";*/newShape + "-14x2";
    }
    else if (originalShape.equals("biped-6x4") && scale == 2) {
      newShape = newShape + "-4x3";
    }
    else if (originalShape.equals("comb-14x2") && scale == 2) {
      newShape = newShape + "-7x2";
    }
    Grid<? extends SensingVoxel> newBody = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(newShape));
    PartiallyDistributedSensing controller = new PartiallyDistributedSensing(newBody, config.contains("none") ? 0 : 1, config.split("-")[0], AbstractPartiallyDistributedMapper.getNumberNeighbors(config.split("-")[0], newBody));
    PartiallyDistributedSensing oldController = ((PartiallyDistributedSensing) ((StepController) originalRobot.getController()).getInnerController());
    for (Grid.Entry<? extends SensingVoxel> entry : newBody) {
      if (entry.getValue() == null) {
        continue;
      }
      controller.getFunctions().set(entry.getX(), entry.getY(), oldController.getFunctions().get(0, 0));
    }
    Grid<? extends SensingVoxel> originalBody = (Grid<? extends SensingVoxel>) originalRobot.getVoxels();
    controller.setDownsamplingParams(scale, Grid.create(originalBody, Objects::nonNull));
    return new Robot<>(new StepController<>(controller, 0.33), newBody);
  }

}
