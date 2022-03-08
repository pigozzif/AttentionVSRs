package world.units.erallab;

import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.Function;


public class BreakVoxels {

  private static final String dir = "/Users/federicopigozzi/Desktop/breakages/";

  public static void main(String[] args) throws IOException {
    BufferedWriter writer = new BufferedWriter(new FileWriter("breakages.csv", false));
    writer.write(String.join(";", "exp", "type", "velocity", "config", "shape", "run") + "\n");
    for (File file : Objects.requireNonNull(new File(dir).listFiles())) {
      if (file.getPath().contains("best") && (file.getPath().contains("4x3") || file.getPath().contains("7x2")) && file.getPath().contains("ga") && !(file.getPath().contains("low") || file.getPath().contains("medium") || file.getPath().contains("high"))) {
        System.out.println(file.getPath());
        breakAndwriteOnFile(file, writer);
      }
    }
    writer.close();
  }

  private static void breakAndwriteOnFile(File file, BufferedWriter writer) throws IOException {
    for (String exp : new String[]{"act", "sens"}) {
      for (String type : new String[]{"zero", "frozen", "random"}) {
        for (int seed = 0; seed < 10; ++seed) {
          Random random = new Random(seed);
          Robot<?> robot = parseIndividualFromFile(file.getPath(), random, String.join("-", "broken", "0.5", String.valueOf(seed), exp, type, "15"), -1);
          Outcome outcome = breakOnTerrain(robot, String.join("-","hilly", "1", "10", String.valueOf(seed)), random);
          String[] fragments = file.getPath().split("\\.");
          writer.write(String.join(";", exp, type, String.valueOf(outcome.getVelocity()),
              fragments[4], fragments[5], fragments[2]) + "\n");
        }
      }
    }
  }

  private static Outcome breakOnTerrain(Robot<?> robot, String terrain, Random random) {
    Function<Robot<?>, Outcome> validationLocomotion = Main.buildLocomotionTask(terrain, 30.0, random, new Settings());
    return validationLocomotion.apply(robot);
  }

  public static Robot<?> parseIndividualFromFile(String fileName, Random random, String transformation, int iteration) {
    List<CSVRecord> records;
    List<String> headers;
    try {
      FileReader reader = new FileReader(fileName);
      CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
      records = csvParser.getRecords();
      headers = csvParser.getHeaderNames();
      reader.close();
    } catch (IOException e) {
      throw new RuntimeException(String.format("Cannot read file: %s", fileName));
    }
    if (!headers.contains("best→solution→serialized")) {
      throw new RuntimeException(String.format("Input file %s does not contain serialization column", fileName));
    }
    SerializationUtils.Mode mode = SerializationUtils.Mode.valueOf(SerializationUtils.Mode.GZIPPED_JSON.name().toUpperCase());
    return RobotUtils.buildRobotTransformation(transformation, random)
        .apply(SerializationUtils.deserialize(records.get((iteration == -1) ? records.size() - 1 : iteration).get("best→solution→serialized"), Robot.class, mode));
  }

}
