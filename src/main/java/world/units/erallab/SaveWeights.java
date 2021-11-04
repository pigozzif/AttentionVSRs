package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.core.snapshots.MLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.SnapshotListener;
import it.units.erallab.hmsrobots.tasks.Task;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
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

    public final double[][][] activations;
    public int step;
    public int dv;
    public int din;

    public AttentionListener(String experiment) {
      this.activations = new double[][][]{};
      this.step = 0;
      this.dv = Integer.parseInt(experiment.split("\\.")[4].split("-")[2]);
      this.din = (experiment.contains("biped")) ? 14 : 15;
    }

    @Override
    public void listen(double t, Snapshot snapshot) {
      if (!(snapshot.getContent() instanceof MLPState)) {
        throw new RuntimeException("Attention listener with non-MLPState content");
      }
      this.activations[this.step] = ((MLPState) snapshot.getContent()).getActivationValues();
      this.step++;
    }

    public double[][][] getData() { return this.activations; }

  }

  private static final String dir = "/Users/federicopigozzi/Desktop/PhD/test/AttentionVSRs/output/attention/";

  public static void main(String[] args) throws IOException {
    BufferedWriter writer = new BufferedWriter(new FileWriter("attention_activations.csv", true));
    for (File file : Files.walk(Paths.get(dir)).filter(p -> Files.isRegularFile(p) && p.toString().contains("best") && p.toString().contains("homo|homo") && p.toString().contains(".ga.") && p.toString().contains("biped")).map(Path::toFile).collect(Collectors.toList())) {
      String path = file.getPath();
      System.out.println(path);
      AttentionListener listener = new AttentionListener(path.split("/")[path.split("/").length - 1]);
      Robot<?> robot = SerializationUtils.deserialize(getSerializedFromFile(file), Robot.class, SerializationUtils.Mode.GZIPPED_JSON);
      Outcome data = getOutcomeFromSimulation(robot, listener);
      writer.write(String.join(";", "true", String.valueOf(data.getVelocity())) + '\n');
      //String experiment = path.split("/")[path.split("/").length - 1];
      //int dk = Integer.parseInt(experiment.split("\\.")[4].split("-")[2]);
      //int din = Integer.parseInt(experiment.split("\\.")[4].split("-")[1]);
      //writeWeightsToFile(robot, dk, din, path.replace("best", "weights"));
    }
    writer.close();
  }

  private static Outcome getOutcomeFromSimulation(Robot<?> robot, SnapshotListener listener) {
    Task<Robot<?>, Outcome> task = new Locomotion(30.0, Locomotion.createTerrain("hilly-1-10-0"), new Settings());
    return task.apply(robot, listener);
  }

  private static Outcome getOutcomeFromSimulation(Robot<?> robot) {
    return getOutcomeFromSimulation(robot, null);
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

  private static void writeActivationsToFile(AttentionListener listener, BufferedWriter writer) throws IOException {
    int t = 0;
    for (double[][] activations : listener.getData()) {
      writer.write(t + ";" + String.join(";", Arrays.stream(activations).flatMapToDouble(Arrays::stream).mapToObj(String::valueOf).toArray(String[]::new)));
      ++t;
    }
  }

}
