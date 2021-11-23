package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.snapshots.MLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.dyn4j.dynamics.Settings;
import org.dyn4j.dynamics.Step;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class SaveWeights {

  public static class AttentionListener implements SnapshotListener {

    public double[][][] activations;
    public int step;

    public AttentionListener(int numVoxels) {
      this.activations = new double[numVoxels][][];
      this.step = 0;
    }

    @Override
    public void listen(double t, Snapshot snapshot) {
      int id = 0;
      for (Grid.Entry<TimedRealFunction> func : ((DistributedSensingShape) snapshot.getChildren().get(1).getChildren().get(0).getContent()).getFunctionGrid()) {
        if (func.getValue() == null) {
          continue;
        }
        this.activations[id++] = ((SelfAttention) func.getValue()).getAttention();
      }
      this.step++;
    }

    public double[][][] getData() { return this.activations; }

  }

  private static final String dir = "/Users/federicopigozzi/Desktop/pos-no-pos-no_messages/";

  public static void main(String[] args) throws IOException {
    BufferedWriter writer = new BufferedWriter(new FileWriter("freeze_attention.csv", true));
    writer.write("t;velocity;config;shape;run\n");
    for (File file : Files.walk(Paths.get(dir)).filter(p -> Files.isRegularFile(p) && p.toString().contains("best") && p.toString().contains("homo|homo") && p.toString().contains(".ga.") && p.toString().contains("attention")).map(Path::toFile).collect(Collectors.toList())) {
      String path = file.getPath();
      System.out.println(path);
      //Robot<?> robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      freezeAttentionToFile(writer, file);
      //Outcome data = getOutcomeFromSimulation(robot, listener);
      //String experiment = path.split("/")[path.split("/").length - 1];
      //int dk = Integer.parseInt(experiment.split("\\.")[4].split("-")[2]);
      //int din = Integer.parseInt(experiment.split("\\.")[4].split("-")[1]);
      //writeWeightsToFile(robot, dk, din, path.replace("best", "weights"));
    }
    writer.close();
  }

  private static Outcome getOutcomeFromSimulation(Robot<?> robot, double t, SnapshotListener listener) {
    Task<Robot<?>, Outcome> task = new Locomotion(t, Locomotion.createTerrain("hilly-1-10-0"), new Settings());
    return task.apply(robot, listener);
  }

  private static Outcome getOutcomeFromSimulation(Robot<?> robot) {
    return getOutcomeFromSimulation(robot, 30.0, null);
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

  private static void writeWeightsToFile(Robot<?> robot, int dk, int din, String fileName) throws IOException {
    BufferedWriter writer = new BufferedWriter(new FileWriter(fileName));
    double[] weights = Arrays.stream(((SelfAttention) ((PartiallyDistributedSensing) ((StepController<? extends SensingVoxel>) robot.getController()).getInnerController()).getFunctions().get(0, 0)).getAttentionParams()).limit(((long) din * dk) * 2).toArray();
    writer.write("t;" + String.join(";", IntStream.range(0, weights.length).mapToObj(i -> "w" + i).toArray(String[]::new)) + "\n");
    writer.write(String.join(";", Arrays.stream(weights).mapToObj(String::valueOf).toArray(String[]::new)));
    writer.close();
  }

  /*private static void writeActivationsToFile(AttentionListener listener, BufferedWriter writer) throws IOException {
    int t = 0;
    for (double[][] activations : listener.getData()) {
      writer.write(t + ";" + String.join(";", Arrays.stream(activations).flatMapToDouble(Arrays::stream).mapToObj(String::valueOf).toArray(String[]::new)));
      ++t;
    }
  }*/

  private static void freezeAttentionToFile(BufferedWriter writer, File file) throws IOException {
    AttentionListener listener = new AttentionListener((file.getPath().contains("biped") ? 10 : 11));
    Robot<?> robot;
    Outcome data;
    for (int i = 1; i < 30; ++i) {
      robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      getOutcomeFromSimulation(robot, i, listener);
      double[][][] attention = listener.getData();
      robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      int id = 0;
      for (Grid.Entry<TimedRealFunction> func : ((PartiallyDistributedSensing) ((StepController) robot.getController()).getInnerController()).getFunctions()) {
        if (func.getValue() == null) {
          continue;
        }
        ((SelfAttention) func.getValue()).setAttention(attention[id++]);
        ((SelfAttention) func.getValue()).freeze();
      }
      data = getOutcomeFromSimulation(robot);
      String[] fragments = file.getPath().split("\\.");
      writer.write(String.join(";", String.valueOf(i), String.valueOf(data.getVelocity()),
              fragments[fragments.length - 3], fragments[fragments.length - 2], fragments[fragments.length - 5]) + "\n");
    }
  }

}
