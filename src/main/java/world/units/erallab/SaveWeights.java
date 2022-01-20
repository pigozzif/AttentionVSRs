package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.stream.Collectors;
import java.util.stream.IntStream;


public class SaveWeights {

  public static class AttentionListener implements SnapshotListener {

    public double[][][][] activations;
    public int step;

    public AttentionListener(int numVoxels, int sec) {
      this.activations = new double[sec * 60 + 1][numVoxels][4][4];
      this.step = 0;
    }

    @Override
    public void listen(double t, Snapshot snapshot) {
      int id = 0;
      DistributedSensingShape shapeSnapshot = ((DistributedSensingShape) snapshot.getChildren().get(1).getChildren().get(0).getContent());
      for (Grid.Entry<TimedRealFunction> func : shapeSnapshot.getFunctionGrid()) {
        if (func.getValue() == null) {
          continue;
        }
        double[][] attention = ((SelfAttention) func.getValue()).getAttention();
        for (int i = 0; i < attention.length; ++i) {
          System.arraycopy(attention[i], 0, this.activations[this.step][id][i], 0, attention[i].length);
        }
        ++id;
      }
      ++this.step;
    }

    public double[][][][] getData() { return this.activations; }

  }

  private static final String dir = "/Users/federicopigozzi/Desktop/pos-no-pos-no_messages/";

  public static void main(String[] args) throws IOException {
    BufferedWriter writer = new BufferedWriter(new FileWriter("freeze_attention.csv", false));
    writer.write("t;attention;config;shape;run\n");
    for (File file : Files.walk(Paths.get(dir)).filter(p -> Files.isRegularFile(p) && p.toString().contains("best") && p.toString().contains("attention") && (p.toString().contains("4x3") || p.toString().contains("7x2"))).map(Path::toFile).collect(Collectors.toList())) {
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
    Robot<?> robot;
    Outcome data;
    for (int i = 1; i < 30; ++i) {
      AttentionListener listener = new AttentionListener((file.getPath().contains("biped") ? 10 : 11), i);
      robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      getOutcomeFromSimulation(robot, i, listener);
      double[][][][] attention = listener.getData();
      robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      int id = 0;
      for (Grid.Entry<TimedRealFunction> func : ((PartiallyDistributedSensing) ((StepController) robot.getController()).getInnerController()).getFunctions()) {
        if (func.getValue() == null) {
          continue;
        }
        ((SelfAttention) func.getValue()).setAttention(attention[attention.length - 1][id++]);
        ((SelfAttention) func.getValue()).freeze();
      }
      data = getOutcomeFromSimulation(robot);
      String[] fragments = file.getPath().split("\\.");
      writer.write(String.join(";", String.valueOf(i), String.valueOf(data.getVelocity()),
              fragments[fragments.length - 3], fragments[fragments.length - 2], fragments[fragments.length - 5]) + "\n");
    }
  }

  private static void saveAttention(BufferedWriter writer, File file) throws IOException {
    AttentionListener listener = new AttentionListener((file.getPath().contains("biped") ? 10 : 11), 30);
    Robot<?> robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
    getOutcomeFromSimulation(robot, 30.0, listener);
    int t = 0;
    for (double[][][] attention : listener.getData()) {
      String attentionString = Arrays.stream(attention).flatMap(Arrays::stream).flatMapToDouble(Arrays::stream).mapToObj(String::valueOf).collect(Collectors.joining("/"));
      String[] fragments = file.getPath().split("\\.");
      writer.write(String.join(";", String.valueOf(t++), attentionString,fragments[fragments.length - 3], fragments[fragments.length - 2], fragments[fragments.length - 5]) + "\n");
    }
  }

}
