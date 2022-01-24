package world.units.erallab;

import it.units.erallab.hmsrobots.core.controllers.TimedRealFunction;
import it.units.erallab.hmsrobots.core.geometry.BoundingBox;
import it.units.erallab.hmsrobots.core.snapshots.MLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Grid;
import it.units.erallab.hmsrobots.viewers.DrawingUtils;
import it.units.erallab.hmsrobots.viewers.drawers.Drawer;

import java.awt.*;
import java.awt.geom.Rectangle2D;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.DoubleStream;


public class AttentionDrawer implements Drawer {

  private final static double LEGEND_COLORS = 15;

  private final Grid<Boolean> body;
  private final Color minColor;
  private final Color zeroColor;
  private final Color maxColor;
  private final Color textColor;
  private final Color axesColor;
  private final double legendSize;
  private final Grid<double[][]> attentionGrid;

  public AttentionDrawer(Grid<Boolean> body, Color minColor, Color zeroColor, Color maxColor, Color textColor, Color axesColor, double legendSize) {
    this.body = body;
    this.minColor = minColor;
    this.zeroColor = zeroColor;
    this.maxColor = maxColor;
    this.textColor = textColor;
    this.axesColor = axesColor;
    this.legendSize = legendSize;
    this.attentionGrid = Grid.create(this.body.getW(), this.body.getH(), (x, y) -> new double[][]{});
  }

  public AttentionDrawer(Grid<Boolean> body) {
    this(body, DrawingUtils.Colors.DATA_NEGATIVE, DrawingUtils.Colors.DATA_ZERO, DrawingUtils.Colors.DATA_POSITIVE, DrawingUtils.Colors.TEXT, DrawingUtils.Colors.AXES, 50);
  }

  @Override
  public void draw(double v, Snapshot snapshot, Graphics2D graphics2D) {
    List<Snapshot> distributedSensingShapes = new ArrayList<>();
    traverse(snapshot, distributedSensingShapes);
    if (distributedSensingShapes.stream().anyMatch(s -> s.getContent() instanceof MLPState)) {
      MLPState state = (MLPState) distributedSensingShapes.stream().filter(s -> s.getContent() instanceof MLPState).findAny().get().getContent();
      for (int i = 0; i < this.body.getW(); ++i) {
        for (int j = 0; j < this.body.getH(); ++j) {
          if (this.body.get(i, j) == null) {
            continue;
          }
          this.attentionGrid.set(i, j, state.getActivationValues());
        }
      }
    }
    else if (distributedSensingShapes.stream().anyMatch(s -> s.getContent() instanceof DistributedSensingShape)) {
      Snapshot state = distributedSensingShapes.stream().filter(s -> s.getContent() instanceof DistributedSensingShape).findAny().get();
      Grid<TimedRealFunction> functions = ((DistributedSensingShape) state.getContent()).getFunctionGrid();
      for (int i = 0; i < this.body.getW(); ++i) {
        for (int j = 0; j < this.body.getH(); ++j) {
          if (functions.get(i, j) == null) {
            continue;
          }
          this.attentionGrid.set(i, j, ((MLPState) ((Snapshottable) functions.get(i, j)).getSnapshot().getContent()).getActivationValues());
        }
      }
    }
    double textH = graphics2D.getFontMetrics().getMaxAscent();
    BoundingBox rBB = BoundingBox.of(
            graphics2D.getClip().getBounds2D().getX(),
            graphics2D.getClip().getBounds2D().getY(),
            graphics2D.getClip().getBounds2D().getMaxX(),
            graphics2D.getClip().getBounds2D().getMaxY()
    );
    /*BoundingBox lBB = BoundingBox.of(
            graphics2D.getClip().getBounds2D().getX(),
            graphics2D.getClip().getBounds2D().getY(),
            graphics2D.getClip().getBounds2D().getMaxX() / 2,
            graphics2D.getClip().getBounds2D().getMaxY()
    );*/
    BoundingBox attentionBB = (rBB.width() > rBB.height()) ? BoundingBox.of(
            rBB.min.x + (rBB.width() - rBB.height()) / 2d + textH,
            rBB.min.y + textH,
            rBB.max.x - (rBB.width() - rBB.height()) / 2d - textH,
            rBB.max.y - textH
    ) : BoundingBox.of(
            rBB.min.x + textH,
            rBB.min.y + (rBB.height() - rBB.width()) / 2d + textH,
            rBB.max.x - textH,
            rBB.max.y - (rBB.height() - rBB.width()) / 2d - textH
    );
    /*BoundingBox robotBB = (lBB.width() > lBB.height()) ? BoundingBox.of(
            lBB.min.x + (lBB.width() - lBB.height()) / 2d + textH,
            lBB.min.y + textH,
            lBB.max.x - (lBB.width() - lBB.height()) / 2d - textH,
            lBB.max.y - textH
    ) : BoundingBox.of(
            lBB.min.x + textH,
            lBB.min.y + (lBB.height() - lBB.width()) / 2d + textH,
            lBB.max.x - textH,
            lBB.max.y - (lBB.height() - lBB.width()) / 2d - textH
    );*/
    this.drawAttention(attentionBB, graphics2D);
    //this.drawRobot(robotBB, graphics2D);
  }

  private static double max(double[][] v) {
    return Arrays.stream(v)
            .mapToDouble(w -> DoubleStream.of(w).max().orElse(0d))
            .max().orElse(0d);
  }

  private static double min(double[][] v) {
    return Arrays.stream(v)
            .mapToDouble(w -> DoubleStream.of(w).min().orElse(0d))
            .min().orElse(0d);
  }

  private void drawAttention(BoundingBox attentionBB, Graphics2D graphics2D) {
    double minX = attentionBB.min.x;
    double wX = attentionBB.width();
    double voxelSizeX = wX / this.body.getW();
    double minY = attentionBB.min.y;
    double max = 1.0;//Double.MIN_VALUE;
    double min = -1.0;//Double.MAX_VALUE;
    /*for (int i = 0; i < this.body.getW(); ++i) {
      for (int j = 0; j < this.body.getH(); ++j) {
        if (!this.body.get(i, j)) {
          continue;
        }
        double[][] attention = this.attentionGrid.get(i, j);
        max = Math.max(max, max(attention));
        min = Math.min(min, min(attention));
      }
    }*/
    for (int i = 0; i < this.body.getW(); ++i) {
      for (int j = 0; j < this.body.getH(); ++j) {
        int local_j = this.body.getH() - j - 1;
        if (!this.body.get(i, j)) {
          continue;
        }
        double startX = minX + voxelSizeX * i;
        double startY = minY + voxelSizeX * local_j;
        graphics2D.setColor(this.axesColor);
        graphics2D.draw(new Rectangle2D.Double(startX, startY, voxelSizeX, voxelSizeX));
        double[][] attention = this.attentionGrid.get(i, j);
        double cellX = voxelSizeX / attention.length;
        for (int n = 0; n < attention.length; ++n) {
          double cellY = voxelSizeX / attention[n].length;
          for (int m = 0; m < attention[n].length; ++m) {
            graphics2D.setColor(DrawingUtils.linear(this.minColor, this.zeroColor, this.maxColor, (float) min, 0, (float) max, (float) attention[n][m]));
            graphics2D.fill(new Rectangle2D.Double(startX + cellX * n, startY + cellY * m, cellX, cellY));
          }
        }
        //cellX /= 2;
        graphics2D.setColor(this.axesColor);
        if (j == body.getH() - 1) {
          float stringCenter = (float) (startX + (graphics2D.getFontMetrics().stringWidth("T") / 2.0));
          float verticalPos = (float) (startY - 1);
          graphics2D.drawString("T", stringCenter, verticalPos);
          stringCenter = (float) (startX + cellX);
          graphics2D.drawString("VX", stringCenter, verticalPos);
          stringCenter = (float) (startX + cellX * 2);
          graphics2D.drawString("VY", stringCenter, verticalPos);
          stringCenter = (float) (startX + (graphics2D.getFontMetrics().stringWidth("A") / 2.0) + cellX * 3);
          graphics2D.drawString("A", stringCenter, verticalPos);
        }
        if (i == 0) {
          float stringCenter = (float) (startX - (graphics2D.getFontMetrics().stringWidth("VX")) - 1);
          float verticalPos = (float) (startY + graphics2D.getFontMetrics().getHeight());
          graphics2D.drawString("T", stringCenter, verticalPos);
          verticalPos += cellX;
          graphics2D.drawString("VX", stringCenter, verticalPos);
          verticalPos += cellX;
          graphics2D.drawString("VY", stringCenter, verticalPos);
          verticalPos += cellX;
          graphics2D.drawString("A", stringCenter, verticalPos);
        }
      }
    }
    this.drawLegend(min, max, BoundingBox.of(minX + wX, minY, minX + wX + this.legendSize, minY + voxelSizeX * this.body.getH()), graphics2D.getFontMetrics().charWidth('m'), graphics2D);
  }

  private void drawRobot(BoundingBox robotBB, Graphics2D graphics2D) {
    double minX = robotBB.min.x;
    double wX = robotBB.width();
    double voxelSizeX = wX / this.body.getW();
    double minY = robotBB.min.y;
    int i = 0 ;
    for (Grid.Entry<Boolean> entry : this.body) {
      if (!entry.getValue()) {
        continue;
      }
      double startX = minX + voxelSizeX * entry.getX();
      double startY = minY + voxelSizeX * (this.body.getH() - entry.getY() - 1);
      graphics2D.setColor(this.axesColor);
      graphics2D.draw(new Rectangle2D.Double(startX, startY, voxelSizeX, voxelSizeX));
      //double[][] posEmbedding = ((SelfAttention) entry.getValue()).getPositionalEmbedding();
      String text = String.valueOf(i);//IntStream.range(0, posEmbedding.length).filter(i -> posEmbedding[i][0] == 1).findFirst().getAsInt());
      double centerX = startX + (voxelSizeX / 2) - (graphics2D.getFontMetrics().stringWidth(text) / 2.0);
      double centerY = startY + (voxelSizeX / 2) + (graphics2D.getFontMetrics().getHeight() / 2.0);
      graphics2D.setColor(this.textColor);
      graphics2D.drawString(text, (int) centerX, (int) centerY);
      ++i;
    }
  }

  private void drawLegend(double min, double max, BoundingBox bb, double textW, Graphics2D g, Color minColor, Color zeroColor, Color maxColor) {
    double deltaY = (bb.max.y - bb.min.y) / LEGEND_COLORS;
    double deltaV = (max - min) / LEGEND_COLORS;
    double colorX = bb.max.x - textW;
    for (int i = 0; i < LEGEND_COLORS; i++) {
      double vMin = min + deltaV * i;
      double vMax = vMin + deltaV;
      double yMin = bb.min.y + deltaY * i;
      double numberHeight = g.getFontMetrics().getHeight() / 2d;
      g.setColor(DrawingUtils.linear(minColor, zeroColor, maxColor, (float) min, 0f, (float) max, (float) vMin));
      g.fill(new Rectangle2D.Double(colorX, yMin, textW, deltaY));
      if (i == 0) {
        g.setColor(textColor);
        String s = String.format("%.1f", vMin);
        g.drawString(s,
                (float) (colorX - textW - g.getFontMetrics().stringWidth(s)),
                (float) (yMin + numberHeight / 2d));
      } else if (vMin <= 0 && vMax >= 0) {
        g.setColor(textColor);
        String s = "0";
        g.drawString(s,
                (float) (colorX - textW - g.getFontMetrics().stringWidth(s)),
                (float) (yMin + deltaY / 2d + numberHeight / 2d));
      } else if (i >= LEGEND_COLORS - 1) {
        g.setColor(textColor);
        String s = String.format("%.1f", vMax);
        g.drawString(s,
                (float) (colorX - textW - g.getFontMetrics().stringWidth(s)),
                (float) (yMin + deltaY + numberHeight / 2d));
      }
    }
  }

  private void drawLegend(double min, double max, BoundingBox bb, double textW, Graphics2D g) {
    drawLegend(min, max, bb, textW, g, minColor, zeroColor, maxColor);
  }

  private static void traverse(Snapshot snapshot, List<Snapshot> targets) {
    //if (snapshot.getContent() instanceof DistributedSensingShape) {
      targets.add(snapshot);
    //}
    for (Snapshot s : snapshot.getChildren()) {
      traverse(s, targets);
    }
  }

}
