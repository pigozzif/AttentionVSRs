
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.Parametrized;
import it.units.erallab.hmsrobots.util.RobotUtils;
import org.junit.Test;
import world.units.erallab.SelfAttention;
import world.units.erallab.mappers.MLPPartiallyDistributedMapper;
import world.units.erallab.PartiallyDistributedSensing;
import world.units.erallab.mappers.SelfAttentionPartiallyDistributedMapper;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static org.junit.Assert.*;


public class testMapper {

    private static final String[] baselineConfigs = new String[] {"none", "neumann", "moore", "all"};
    private static final int[] target = new int[] {20 + 5, 72 + 18, 156 + (12 * 2) + 2, ((14 * 14) + 14) + ((14 * 2) + 2)};
    private static final String attentionConfig = "all-14-2-2";
    private static final int attentionTarget = 6 + 28 * 3;
    private static final int downstreamTarget = ((20 * 2) + 2);

    private static Grid<? extends SensingVoxel> getTestBody() {
        return RobotUtils.buildSensorizingFunction("uniform-a+vxy+t-0.01").apply(RobotUtils.buildShape("biped-4x3"));
    }

    private static List<Double> getTestHomo(int size) {
        return IntStream.range(0, size).mapToDouble(i -> 1.0).boxed().collect(Collectors.toList());
    }

    private static List<Double> getTestHetero(int size) {
        return IntStream.range(0, size).mapToDouble(i -> 0.0).boxed().collect(Collectors.toList());
    }

    @Test
    public void testGenotypeSizeBaselineHomo() {
        for (int i = 0; i < baselineConfigs.length; ++i) {
            MLPPartiallyDistributedMapper test = new MLPPartiallyDistributedMapper(getTestBody(), baselineConfigs[i] + "-homo");
            assertEquals(target[i], test.getGenotypeSize());
        }
    }

    @Test
    public void testApplyBaselineHomo() {
        for (String baselineConfig : baselineConfigs) {
            MLPPartiallyDistributedMapper test = new MLPPartiallyDistributedMapper(getTestBody(), baselineConfig + "-homo");
            List<Double> gen = getTestHomo(test.getGenotypeSize());
            assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || Arrays.equals(((Parametrized) f.getValue()).getParams(), gen.stream().mapToDouble(d -> d).toArray())));
        }
    }

    @Test
    public void testGenotypeSizeBaselineHetero() {
        for (int i = 0; i < baselineConfigs.length; ++i) {
            MLPPartiallyDistributedMapper test = new MLPPartiallyDistributedMapper(getTestBody(), baselineConfigs[i] + "-hetero");
            assertEquals(target[i] * 10, test.getGenotypeSize());
        }
    }

    @Test
    public void testApplyBaselineHetero() {
        for (String baselineConfig : baselineConfigs) {
            MLPPartiallyDistributedMapper test = new MLPPartiallyDistributedMapper(getTestBody(), baselineConfig + "-hetero");
            List<Double> gen = getTestHetero(test.getGenotypeSize());
            assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || Arrays.equals(((Parametrized) f.getValue()).getParams(), gen.subList(0, test.getGenotypeSizeForVoxel()).stream().mapToDouble(d -> d).toArray())));
        }
    }

    @Test
    public void testGenotypeSizeAttentionHomoHomo() {
        String config = attentionConfig + "-homo|homo";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        assertEquals(attentionTarget + downstreamTarget, test.getGenotypeSize());
    }

    @Test
    public void testApplyAttentionHomoHomo() {
        String config = attentionConfig + "-homo|homo";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        List<Double> gen = getTestHomo(test.getGenotypeSize());
        assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || Arrays.equals(((Parametrized) f.getValue()).getParams(), gen.stream().mapToDouble(d -> d).toArray())));
    }

    @Test
    public void testGenotypeSizeAttentionHeteroHetero() {
        String config = attentionConfig + "-hetero|hetero";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        assertEquals((attentionTarget + downstreamTarget) * 10, test.getGenotypeSize());
    }

    @Test
    public void testApplyAttentionHeteroHetero() {
        String config = attentionConfig + "-hetero|hetero";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        List<Double> gen = getTestHetero(test.getGenotypeSize());
        assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || Arrays.equals(((Parametrized) f.getValue()).getParams(), gen.subList(0, test.getGenotypeSizeForVoxel()).stream().mapToDouble(d -> d).toArray())));
    }

    @Test
    public void testGenotypeSizeAttentionHeteroHomo() {
        String config = attentionConfig + "-hetero|homo";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        assertEquals((attentionTarget * 10) + downstreamTarget, test.getGenotypeSize());
    }

    @Test
    public void testApplyAttentionHeteroHomo() {
        String config = attentionConfig + "-hetero|homo";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        List<Double> downGen = getTestHomo(test.getDownstreamSizeForVoxel());
        List<Double> attGen = getTestHetero(test.getAttentionSizeForVoxel() * 10);
        List<Double> gen = getTestHomo(test.getDownstreamSizeForVoxel());
        gen.addAll(getTestHetero(test.getAttentionSizeForVoxel() * 10));
        assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || (Arrays.equals(((SelfAttention) f.getValue()).getAttentionParams(), attGen.subList(0, test.getAttentionSizeForVoxel()).stream().mapToDouble(d -> d).toArray()) &&
                Arrays.equals(((SelfAttention) f.getValue()).getDownstreamParams(), downGen.stream().mapToDouble(d -> d).toArray()))));
    }

    @Test
    public void testGenotypeSizeAttentionHomoHetero() {
        String config = attentionConfig + "-homo|hetero";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        assertEquals(attentionTarget + (downstreamTarget * 10), test.getGenotypeSize());
    }

    @Test
    public void testApplyAttentionHomoHetero() {
        String config = attentionConfig + "-homo|hetero";
        SelfAttentionPartiallyDistributedMapper test = new SelfAttentionPartiallyDistributedMapper(getTestBody(), config);
        List<Double> downGen = getTestHetero(test.getDownstreamSizeForVoxel() * 10);
        List<Double> attGen = getTestHomo(test.getAttentionSizeForVoxel());
        List<Double> gen = getTestHomo(test.getAttentionSizeForVoxel());
        gen.addAll(getTestHetero(test.getDownstreamSizeForVoxel() * 10));
        assertTrue(((PartiallyDistributedSensing) test.apply(gen).getController()).getFunctions().stream().allMatch(f -> f.getValue() == null || (Arrays.equals(((SelfAttention) f.getValue()).getAttentionParams(), attGen.stream().mapToDouble(d -> d).toArray()) &&
                Arrays.equals(((SelfAttention) f.getValue()).getDownstreamParams(), downGen.subList(0, test.getDownstreamSizeForVoxel()).stream().mapToDouble(d -> d).toArray()))));
    }

}
