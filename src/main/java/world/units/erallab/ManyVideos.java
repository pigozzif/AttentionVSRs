package world.units.erallab;

import java.io.File;
import java.io.IOException;
import java.util.Objects;


public class ManyVideos {

  private static final String dir = "/Users/federicopigozzi/Desktop/pos-no-pos-no_messages/";

  public static void main(String[] args) throws IOException {
    for (File file : Objects.requireNonNull(new File(dir).listFiles())) {
      if (file.getPath().contains("best") && file.getPath().contains("attention") && !(file.getPath().contains("low") || file.getPath().contains("medium") || file.getPath().contains("high"))) {
        System.out.println(file.getPath());
        String[] fragments = file.getPath().split("\\.");
        String shape = fragments[fragments.length - 2].replace("14x2", "large").replace("7x2", "").replace("4x3", "").replace("6x4", "large");
        VideoMaker.main(new String[]{"input=" + file.getPath(), "output=" + String.join("-", "best", shape, fragments[fragments.length - 5]) + ".mp4"});
      }
    }
  }

}
