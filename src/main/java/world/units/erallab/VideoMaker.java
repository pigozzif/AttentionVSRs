/*
 * Copyright 2020 Eric Medvet <eric.medvet@gmail.com> (as eric)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package world.units.erallab;

import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.erallab.hmsrobots.viewers.*;
import it.units.erallab.hmsrobots.viewers.drawers.*;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.apache.commons.lang3.tuple.Pair;
import org.dyn4j.dynamics.Settings;

import java.io.*;
import java.util.*;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.ScheduledExecutorService;
import java.util.logging.Logger;
import java.util.stream.Collectors;

import static it.units.malelab.jgea.core.util.Args.*;

/**
 * @author eric
 */
public class VideoMaker {

    private static final Logger L = Logger.getLogger(VideoMaker.class.getName());

    public static void main(String[] args) throws IOException, InterruptedException {
        //get params
        String inputFileName = a(args, "input", null);
        int numDirs = inputFileName.split("/").length;
        String seed = inputFileName.split("/")[numDirs - 1].split("\\.")[2];
        String outputFileName = a(args, "output", null);
        String serializedRobotColumn = a(args, "serializedRobotColumnName", "serialized");
        String terrainName = a(args, "terrain", "hilly-1-10-" + seed);
        String transformationName = a(args, "transformation", "identity");
        double startTime = d(a(args, "startTime", "0.0"));
        double endTime = d(a(args, "endTime", "30.0"));
        int w = i(a(args, "w", "900"));
        int h = i(a(args, "h", "600"));
        int frameRate = i(a(args, "frameRate", "30"));
        String encoderName = a(args, "encoder", VideoUtils.EncoderFacility.FFMPEG_LARGE.name());
        SerializationUtils.Mode mode = SerializationUtils.Mode.valueOf(a(args, "deserializationMode", SerializationUtils.Mode.GZIPPED_JSON.name()).toUpperCase());
        //read data
        Reader reader = null;
        List<CSVRecord> records = null;
        List<String> headers = null;
        parseBestFromFile(inputFileName);
        String input = "./to_film.txt";
        try {
            reader = new FileReader(input);
            CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
            records = csvParser.getRecords();
            headers = csvParser.getHeaderNames();
            reader.close();
        } catch (IOException e) {
            L.severe(String.format("Cannot read input data: %s", e));
            if (reader != null) {
                try {
                    reader.close();
                } catch (IOException ioException) {
                    //ignore
                }
            }
            System.exit(-1);
        }
        L.info(String.format("Read %d data lines from %s with columns %s",
                records.size(),
                input,
                headers
        ));
        //check columns
        if (headers.size() < 3) {
            L.severe(String.format("Found %d columns: expected 3 or more", headers.size()));
            System.exit(-1);
        }
        if (!headers.contains(serializedRobotColumn)) {
            L.severe(String.format("Cannot find serialized robot column %s in %s", serializedRobotColumn, headers));
            System.exit(-1);
        }
        //find x- and y- values
        String xHeader = headers.get(0);
        String yHeader = headers.get(1);
        List<String> xValues = records.stream()
                .map(r -> r.get(xHeader))
                .distinct()
                .collect(Collectors.toList());
        List<String> yValues = records.stream()
                .map(r -> r.get(yHeader))
                .distinct()
                .collect(Collectors.toList());
        //build grid
        List<CSVRecord> finalRecords = records;
        Grid<List<String>> rawGrid = Grid.create(
                xValues.size(),
                yValues.size(),
                (x, y) -> finalRecords.stream()
                        .filter(r -> r.get(xHeader).equals(xValues.get(x)) && r.get(yHeader).equals(yValues.get(y)))
                        .map(r -> r.get(serializedRobotColumn))
                        .collect(Collectors.toList())
        );
        //build named grid of robots
        Grid<Pair<String, Robot<?>>> namedRobotGrid = Grid.create(
                rawGrid.getW(),
                rawGrid.getH(),
                (x, y) -> rawGrid.get(x, y).isEmpty() ? null : Pair.of(
                        xValues.get(x) + " " + yValues.get(y),
                        RobotUtils.buildRobotTransformation(transformationName, new Random(0))
                                .apply(SerializationUtils.deserialize(rawGrid.get(x, y).get(0), Robot.class, mode))
                )
        );
        //prepare problem
        Locomotion locomotion = new Locomotion(endTime, Locomotion.createTerrain(terrainName), new Settings());
        //do simulations
        ScheduledExecutorService uiExecutor = Executors.newScheduledThreadPool(4);
        ExecutorService executor = Executors.newCachedThreadPool();
        GridSnapshotListener gridSnapshotListener = null;
        if (outputFileName == null) {
            gridSnapshotListener = new GridOnlineViewer(
                    Grid.create(namedRobotGrid, p -> p == null ? null : p.getLeft()),
                    Grid.create(namedRobotGrid, p -> p == null ? null : (!inputFileName.contains("baseline")) ? getAttentionDrawer(p.getLeft(), inputFileName) : Drawers.basicWithMiniWorldAndBrain(p.getLeft())),//Drawers.basicWithMiniWorldAndBrain(p.getLeft())),
                    uiExecutor
            );
            ((GridOnlineViewer) gridSnapshotListener).start(3);
        } else {
            try {
                gridSnapshotListener = new GridFileWriter(
                        w, h, startTime, frameRate, VideoUtils.EncoderFacility.valueOf(encoderName.toUpperCase()),
                        new File(outputFileName),
                        Grid.create(namedRobotGrid, p -> p == null ? null : p.getLeft()),
                        Grid.create(namedRobotGrid, p -> p == null ? null : (!inputFileName.contains("baseline")) ? getAttentionDrawer(p.getLeft(), inputFileName) : Drawers.basicWithMiniWorld(p.getLeft())));
            } catch (IOException e) {
                L.severe(String.format("Cannot build grid file writer: %s", e));
                System.exit(-1);
            }
        }
        GridEpisodeRunner<Robot<?>> runner = new GridEpisodeRunner<>(
                namedRobotGrid,
                locomotion,
                gridSnapshotListener,
                executor
        );
        runner.run();
        if (outputFileName != null) {
            executor.shutdownNow();
            uiExecutor.shutdownNow();
        }
    }

    public static void parseBestFromFile(String file) throws IOException {
        BufferedWriter writer = new BufferedWriter(new FileWriter("to_film.txt"));
        Reader reader = new FileReader(file);
        List<CSVRecord> records;
        CSVParser csvParser = CSVFormat.DEFAULT.withDelimiter(';').withFirstRecordAsHeader().parse(reader);
        records = csvParser.getRecords();
        String best = records.get(records.size() - 1).get("best→solution→serialized");
        reader.close();
        writer.write("x;y;serialized\n");
        writer.write("0;0;" + best);
        writer.close();
    }

    public static Drawer getAttentionDrawer(String string, String name) {
        String shape = name.split("\\.")[5];
        return Drawer.of(
                Drawer.clip(
                        BoundingBox.of(0d, 0d, 1d, 0.5d),
                        Drawers.basicWithMiniWorld()
                ),
                Drawer.clip(
                        BoundingBox.of(0d, 0.5d, 1d, 1d),
                        Drawer.of(
                                Drawer.clear(),
                                new AttentionDrawer(RobotUtils.buildShape(shape))
                        )
                ),
                new InfoDrawer(string)
        );
    }

}
