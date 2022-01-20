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


public class SurrogateValidator {

  private static final String[] terrains = {"flat", "hilly-1-10-0", "hilly-1-10-1", "hilly-1-10-2", "hilly-1-10-3", "hilly-1-10-4",
          "steppy-1-10-0", "steppy-1-10-1", "steppy-1-10-2", "steppy-1-10-3", "steppy-1-10-4", "uphill-10", "uphill-20", "downhill-10", "downhill-20"};
  private static final String[] header = {"validation.terrain", "validation.transformation", "validation.seed",
          "outcome.computation.time", "outcome.distance", "outcome.velocity", "\n"};
  private static final String dir = /*System.getProperty("user.dir") + "/output/";*/ "/Users/federicopigozzi/Desktop/new_seeds/";

  public static void main(String[] args) throws IOException {
    for (File file : Objects.requireNonNull(new File(dir).listFiles())) {
      if (file.getPath().contains("best") && (file.getPath().contains("4x3") || file.getPath().contains("7x2"))) {
        System.out.println(file.getPath());
        validateAndwriteOnFile(file);
      }
    }
  }

  private static void validateAndwriteOnFile(File file) throws IOException {
    String validationFile = file.getPath().replace("best", "validation");
    BufferedWriter writer = new BufferedWriter(new FileWriter(validationFile));
    writer.write(String.join(";", header));
    String path = file.getPath().split("/")[file.getPath().split("/").length - 1];
    int seed = Integer.parseInt(path.split("\\.")[2]);
    Random random = new Random(seed);
    for (String terrain : terrains) {
      Robot<?> robot = parseIndividualFromFile(file.getPath(), random, -1);
      Outcome outcome = validateOnTerrain(robot, terrain, random);
      writer.write(String.join(";", terrain, "identity", String.valueOf(seed),
              String.valueOf(outcome.getComputationTime()), String.valueOf(outcome.getDistance()),
              String.valueOf(outcome.getVelocity()), "\n"));
    }
    writer.close();
  }

  private static Outcome validateOnTerrain(Robot<?> robot, String terrain, Random random) {
    Function<Robot<?>, Outcome> validationLocomotion = Main.buildLocomotionTask(terrain, 30.0, random, new Settings());
    return validationLocomotion.apply(robot);
  }

  public static Robot<?> parseIndividualFromFile(String fileName, Random random, int iteration) {
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
    return RobotUtils.buildRobotTransformation("identity", random)
            .apply(SerializationUtils.deserialize(records.get((iteration == -1) ? records.size() - 1 : iteration).get("best→solution→serialized"), Robot.class, mode));
  }

}
