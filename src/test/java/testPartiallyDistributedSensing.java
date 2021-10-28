import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Locomotion;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.RobotUtils;
import org.apache.commons.math3.util.Pair;
import org.dyn4j.dynamics.Settings;
import org.junit.Test;
import world.units.erallab.PartiallyDistributedSensing;
import world.units.erallab.PartiallyDistributedSensingMultiple;

import java.util.List;
import java.util.Objects;
import java.util.Random;
import java.util.function.BiFunction;
import java.util.function.Function;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.*;


public class testPartiallyDistributedSensing {

    private static Robot<?> getTestRobot(BiFunction<Pair<Integer, Integer>, Grid<? extends SensingVoxel>, List<Pair<Integer, Integer>>> neighborhood, int nNeighbors, String config) {
        Grid<? extends SensingVoxel> body = RobotUtils.buildSensorizingFunction("empty").apply(RobotUtils.buildShape("worm-4x3"));
        PartiallyDistributedSensing controller = new PartiallyDistributedSensing(body, (nNeighbors == 0) ? 0 : 1, config, nNeighbors);
        for (Grid.Entry<? extends SensingVoxel> entry : body) {
            if (entry.getValue() == null) {
                continue;
            }
            MultiLayerPerceptron mlp = new MultiLayerPerceptron(
                    MultiLayerPerceptron.ActivationFunction.TANH,
                    controller.nOfInputs(entry.getX(), entry.getY()),
                    new int[]{},
                    controller.nOfOutputs(entry.getX(), entry.getY())
            );
            double[] ws = mlp.getParams();
            IntStream.range(0, ws.length).forEach(i -> ws[i] = (new Random(0)).nextDouble() * 2d - 1d);
            mlp.setParams(ws);
            controller.getFunctions().set(entry.getX(), entry.getY(), mlp);
        }
        return new Robot<>(controller, body);
    }

    private static List<Robot<?>> getTestRobots() {
        return List.of(getTestRobot((x, y) -> List.of(), 0, "none"),
                getTestRobot((x, y) -> List.of(new Pair<>(x.getFirst() + 1, x.getSecond()), new Pair<>(x.getFirst(), x.getSecond() + 1), new Pair<>(x.getFirst() - 1, x.getSecond()), new Pair<>(x.getFirst(), x.getSecond() - 1)), 4, "neumann"),
                getTestRobot((x, y) -> List.of(new Pair<>(x.getFirst() + 1, x.getSecond() + 1), new Pair<>(x.getFirst() - 1, x.getSecond() - 1), new Pair<>(x.getFirst() - 1, x.getSecond() + 1), new Pair<>(x.getFirst() + 1, x.getSecond() - 1), new Pair<>(x.getFirst() + 1, x.getSecond()), new Pair<>(x.getFirst(), x.getSecond() + 1), new Pair<>(x.getFirst() - 1, x.getSecond()), new Pair<>(x.getFirst(), x.getSecond() - 1)), 8, "moore"),
                getTestRobot((x, y) -> y.stream().filter(v -> v.getValue() != null).map(e -> new Pair<>(e.getX(), e.getY())).collect(Collectors.toList()), 12, "all"));
    }

    @Test
    public void testFunctions() {
        assertArrayEquals(new int[] {12, 12, 12, 12}, getTestRobots().stream().mapToInt(robot -> (int) ((PartiallyDistributedSensing) robot.getController()).getFunctions().count(Objects::nonNull)).toArray());
    }

    @Test
    public void testInputs() {
        int[] target = new int[] {0, 4, 8, 12};
        List<Robot<?>> robots = getTestRobots();
        assertArrayEquals(new int[] {12, 12, 12, 12}, IntStream.range(0, robots.size()).map(i -> (int) ((PartiallyDistributedSensing) robots.get(i).getController()).getInputGrid().count(e -> e == target[i])).toArray());
    }

    @Test
    public void testLastSignals() {
        int[] target = new int[] {0, 1, 1, 1};
        List<Robot<?>> robots = getTestRobots();
        assertArrayEquals(new int[] {12, 12, 12, 12}, IntStream.range(0, robots.size()).map(i -> (int) ((PartiallyDistributedSensing) robots.get(i).getController()).getLastSignalsGrid().count(e -> e.length == target[i])).toArray());
    }

    @Test(expected=Test.None.class)
    public void testExecution() {
        Function<Robot<?>, Outcome> trainingTask = new Locomotion(60.0, Locomotion.createTerrain("flat"), new Settings());
        getTestRobots().forEach(trainingTask::apply);
    }

}
