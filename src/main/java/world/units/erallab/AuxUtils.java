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

import it.units.erallab.hmsrobots.core.controllers.AbstractController;
import it.units.erallab.hmsrobots.core.controllers.StepController;
import it.units.erallab.hmsrobots.core.objects.Robot;
import it.units.erallab.hmsrobots.core.objects.SensingVoxel;
import it.units.erallab.hmsrobots.tasks.locomotion.Outcome;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.util.SerializationUtils;
import it.units.malelab.jgea.core.Individual;
import it.units.malelab.jgea.core.evolver.Event;
import it.units.malelab.jgea.core.listener.NamedFunction;
import it.units.malelab.jgea.core.listener.NamedFunctions;
import it.units.malelab.jgea.core.util.*;

import java.util.*;
import java.util.function.Function;
import java.util.function.Predicate;
import java.util.stream.Collectors;
import java.util.stream.IntStream;

import static it.units.malelab.jgea.core.listener.NamedFunctions.*;

/**
 * @author eric
 */
public class AuxUtils {

    private AuxUtils() {
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> basicFunctions() {
        return List.of(
                iterations(),
                births(),
                fitnessEvaluations(),
                elapsedSeconds()
        );
    }

    public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> serializationFunction(boolean flag) {
        if (!flag) {
            return List.of();
        }
        return List.of(f("serialized", r -> SerializationUtils.serialize(r, SerializationUtils.Mode.GZIPPED_JSON)).of(solution()));
    }

    public static List<NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?>> individualFunctions(Function<Outcome, Double> fitnessFunction) {
        NamedFunction<Individual<?, ? extends Robot<?>, ? extends Outcome>, ?> size = size().of(genotype());
        return List.of(
                f("w", "%2d", (Function<Grid<?>, Number>) Grid::getW)
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                f("h", "%2d", (Function<Grid<?>, Number>) Grid::getH)
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                f("num.voxel", "%2d", (Function<Grid<?>, Number>) g -> g.count(Objects::nonNull))
                        .of(f("shape", (Function<Robot<?>, Grid<?>>) Robot::getVoxels))
                        .of(solution()),
                size.reformat("%5d"),
                genotypeBirthIteration(),
                f("fitness", "%5.1f", fitnessFunction).of(fitness()),
                f("uniformity", "%5.1f", s -> {
                  AbstractController<? extends SensingVoxel> c = ((StepController) ((Robot<?>) s).getController()).getInnerController();
                  if (c instanceof PartiallyDistributedSensing) {
                    return ((PartiallyDistributedSensing) c).getUniformity();
                  }
                  else {
                    return -1.0;
                  }
                }).of(solution())
        );
    }

    public static List<NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?>> populationFunctions(Function<Outcome, Double> fitnessFunction) {
        NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?> min = min(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
        NamedFunction<Event<?, ? extends Robot<?>, ? extends Outcome>, ?> median = median(Double::compare).of(each(f("fitness", fitnessFunction).of(fitness()))).of(all());
        return List.of(
                size().of(all()),
                size().of(firsts()),
                size().of(lasts()),
                uniqueness().of(each(genotype())).of(all()),
                uniqueness().of(each(solution())).of(all()),
                uniqueness().of(each(fitness())).of(all()),
                min.reformat("%+4.1f"),
                median.reformat("%5.1f")
        );
    }

    public static List<NamedFunction<Outcome, ?>> basicOutcomeFunctions() {
        return List.of(
                f("computation.time", "%4.2f", Outcome::getComputationTime),
                f("distance", "%5.1f", Outcome::getDistance),
                f("velocity", "%5.1f", Outcome::getVelocity),
                f("corrected.efficiency", "%5.2f", Outcome::getCorrectedEfficiency),
                f("area.ratio.power", "%5.1f", Outcome::getAreaRatioPower),
                f("control.power", "%5.1f", Outcome::getControlPower),
                f("shape.dynamic", "%s", o -> Grid.toString(o.getAveragePosture(8), (Predicate<Boolean>) b -> b,"|"))
        );
    }

    public static List<NamedFunction<Outcome, ?>> detailedOutcomeFunctions(double spectrumMinFreq, double spectrumMaxFreq, int spectrumSize) {
        return Misc.concat(List.of(
                NamedFunction.then(cachedF(
                        "center.x.spectrum",
                        (Outcome o) -> new ArrayList<>(o.getCenterXVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
                        ),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                ),
                NamedFunction.then(cachedF(
                        "center.y.spectrum",
                        (Outcome o) -> new ArrayList<>(o.getCenterYVelocitySpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
                        ),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                ),
                NamedFunction.then(cachedF(
                        "center.angle.spectrum",
                        (Outcome o) -> new ArrayList<>(o.getCenterAngleSpectrum(spectrumMinFreq, spectrumMaxFreq, spectrumSize).values())
                        ),
                        IntStream.range(0, spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                ),
                NamedFunction.then(cachedF(
                        "footprints.spectra",
                        (Outcome o) -> o.getFootprintsSpectra(4, spectrumMinFreq, spectrumMaxFreq, spectrumSize).stream()
                                .map(SortedMap::values)
                                .flatMap(Collection::stream)
                                .collect(Collectors.toList())
                        ),
                        IntStream.range(0, 4 * spectrumSize).mapToObj(NamedFunctions::nth).collect(Collectors.toList())
                )
        ));
    }

}