package world.units.erallab;


import it.units.erallab.hmsrobots.core.controllers.Controller;
import it.units.erallab.hmsrobots.core.controllers.StepController;
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

  private static final String dir = "/Users/federicopigozzi/Desktop/pos-no-pos-no_messages";
  private static final String sensorConfig = "uniform-a+vxy+t-0.01";

  public static void main(String[] args) throws IOException {
    int scale = Integer.parseInt(Args.a(args, "scale", null));
    BufferedWriter writer = new BufferedWriter(new FileWriter("transfer.csv", true));
    for (File file : Files.walk(Paths.get(dir)).filter(p -> Files.isRegularFile(p) && p.toString().contains("best") && p.toString().contains("homo|homo") && p.toString().contains(".ga.")).map(Path::toFile).collect(Collectors.toList())) {
      String path = file.getPath();
      System.out.println(path);
      String seed = path.split("\\.")[2];
      String shape = path.split("\\.")[5];
      String config = path.split("\\.")[4];
      Robot<?> originalRobot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      Robot<?> newRobot = getNewRobot(originalRobot, shape, config, scale);
      Outcome data = getOutcomeFromSimulation(newRobot, seed);
      writer.write(String.join(";", String.valueOf(data.getVelocity()), String.valueOf(scale), config, shape, seed) + "\n");
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
    if (originalShape.contains("biped") && scale == 2) {
      newShape = newShape + "-6x4";
    }
    else if (originalShape.contains("comb") && scale == 2) {
      newShape = newShape + "-14x2";
    }
    Grid<? extends SensingVoxel> newBody = RobotUtils.buildSensorizingFunction(sensorConfig).apply(RobotUtils.buildShape(newShape));
    PartiallyDistributedSensing controller = new PartiallyDistributedSensing(newBody, config.contains("none") ? 0 : 1, config.split("-")[0], AbstractPartiallyDistributedMapper.getNumberNeighbors(config.split("-")[0], newBody));
    controller.setDownsamplingParams(scale, (int) originalRobot.getVoxels().count(Objects::nonNull));
    return new Robot<>(Controller.step(controller, 0.33), newBody);
  }

}
