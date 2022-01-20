package world.units.erallab;

import com.fasterxml.jackson.annotation.JsonCreator;
import com.fasterxml.jackson.annotation.JsonProperty;
import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import it.units.erallab.hmsrobots.core.controllers.RealFunction;
import it.units.erallab.hmsrobots.core.snapshots.MLPState;
import it.units.erallab.hmsrobots.core.snapshots.Snapshot;
import it.units.erallab.hmsrobots.core.snapshots.Snapshottable;
import it.units.erallab.hmsrobots.util.Domain;
import it.units.erallab.hmsrobots.util.Parametrized;
import org.apache.commons.lang3.ArrayUtils;

import java.io.Serializable;
import java.util.Arrays;


public class SelfAttention implements Serializable, Parametrized, RealFunction, Snapshottable {

  @JsonProperty
  private final MultiLayerPerceptron inner;
  @JsonProperty
  private final int n;
  @JsonProperty
  private final int din;
  @JsonProperty
  private final int dk;
  @JsonProperty
  private final int dv;
  @JsonProperty
  private final double[][] wq;
  @JsonProperty
  private final double[][] wk;
  @JsonProperty
  private final double[][] wv;
  @JsonProperty
  private final double[] qbias;
  @JsonProperty
  private final double[] kbias;
  @JsonProperty
  private final double[] vbias;

  private double[][] attention;
  private final double[][] latentCode;
  private final double[][] q;
  private final double[][] k;
  private final double[][] v;
  private boolean freeze;
  private int id;
  private int steps;
  private int t;

  @JsonCreator
  public SelfAttention(@JsonProperty("inner") MultiLayerPerceptron inner,
                       @JsonProperty("n") int n,
                       @JsonProperty("din") int din,
                       @JsonProperty("dk") int dk,
                       @JsonProperty("dv") int dv,
                       @JsonProperty("wq") double[][] wq,
                       @JsonProperty("wk") double[][] wk,
                       @JsonProperty("wv") double[][] wv,
                       @JsonProperty("qbias") double[] qbias,
                       @JsonProperty("kbias") double[] kbias,
                       @JsonProperty("vbias") double[] vbias) {
    this.inner = inner;
    this.n = n;
    this.din = din;
    this.dk = dk;
    this.dv = dv;
    this.wq = wq;
    this.wk = wk;
    this.wv = wv;
    this.qbias = qbias;
    this.kbias = kbias;
    this.vbias = vbias;
    this.attention = new double[din][din];
    this.latentCode = new double[din][n];
    this.q = new double[din][dk];
    this.k = new double[din][dk];
    this.v = new double[din][dv];
    this.freeze = false;
    this.id = -1;
    this.steps = 1;
    this.t = 0;
  }

  public SelfAttention(MultiLayerPerceptron inner, int n, int din, int dk, int dv) {
    this(inner, n, din, dk, dv, new double[1][dk], new double[1][dk], new double[n][dv],
            new double[dk], new double[dk], new double[dv]);
  }
  // TODO: rename
  public static int countParams(int din, int dk, int dv, int n) {
    return countQueriesAndKeysParams(din, dk) + countValuesParams(dv, n);
  }

  public static int countQueriesAndKeysParams(int din, int dk) {
    return (1 * dk) + dk + (1 * dk) + dk;
  }

  public static int countValuesParams(int dv, int n) {
    return (n * dv) + dv;
  }

  public int countParams() { return countParams(this.din, this.dk, this.dv, this.n); }

  public double[] getAttentionParams() { return concat(flat(this.wq), flat(this.wk), flat(this.wv), this.qbias, this.kbias, this.vbias); }

  public double[] getQueriesAndKeysMatrices() { return concat(flat(this.wq), flat(this.wk)); }

  public double[] getQueriesAndKeysBias() { return concat(this.qbias, this.kbias); }

  public double[] getDownstreamParams() { return this.inner.getParams(); }

  @Override
  public double[] getParams() { return concat(this.getAttentionParams(), this.getDownstreamParams()); }

  public void setAttentionParams(double[] params) {
    int s = 0;
    for (double[] row : this.wq) {
      System.arraycopy(params, s, row, 0, this.dk);
      s = s + this.dk;
    }
    System.arraycopy(params, s, this.qbias, 0, this.dk);
    s = s + this.dk;
    for (double[] row : this.wk) {
      System.arraycopy(params, s, row, 0, this.dk);
      s = s + this.dk;
    }
    System.arraycopy(params, s, this.kbias, 0, this.dk);
    s = s + this.dk;
    for (double[] row : this.wv) {
      System.arraycopy(params, s, row, 0, this.dv);
      s = s + this.dv;
    }
    System.arraycopy(params, s, this.vbias, 0, this.dv);
  }

  public void setDownstreamParams(double[] params) { this.inner.setParams(params); }

  @Override
  public void setParams(double[] params) {
    this.setAttentionParams(Arrays.stream(params).limit(countParams(this.din, this.dk, this.dv, this.n)).toArray());
    this.setDownstreamParams(Arrays.stream(params).skip(countParams(this.din, this.dk, this.dv, this.n)).toArray());
  }

  public static double[] concat(double[]... arrays) {
    double[] values = new double[]{};
    for (double[] a : arrays) {
      values = ArrayUtils.addAll(values, a);
    }
    return values;
  }

  @Override
  public double[] apply(double[] inputs) {
    //this.freeze = this.t % this.steps != 0;
    ++this.t;
    return this.inner.apply(flat(this.applyAttention(inputs)));
  }

  public double[][] applyAttention(double[] inputs) {
    double[][] reshaped = reshapeVector(inputs, this.n, this.din);
    int k = 0;
    for (int i = 0; i < this.n; ++i) {
      if (Arrays.stream(reshaped[i]).anyMatch(v -> v != 0.0)) {
        k = i;
        break;
      }
    }
    double[][] originalInputs = reshapeVector(reshaped[k], this.din, 1);
    if (!this.freeze) {
      linearTransform(originalInputs, this.wq, this.qbias, this.q);
      double[][] keys = matrixTranspose(linearTransform(originalInputs, this.wk, this.kbias, this.k));
      //linearTransform(matrixTranspose(reshaped), this.wv, this.vbias, this.v);
      matrixMult(this.q, keys, this.attention);
      matrixDiv(this.attention, Math.sqrt(this.dk));
      for (double[] row : this.attention) {
        tanh(row);//softmax(row);
      }
    }
    matrixMult(this.attention, matrixTranspose(reshaped), this.latentCode);
    //matrixDiv(this.latentCode, Math.sqrt(this.din));
    return this.latentCode;
  }

  public static double[][] reshapeVector(double[] v, int p, int n) {
    if (v.length != p * n) {
      throw new RuntimeException(String.format("Cannot reshape vector of size %d into (%d,%d)", v.length, p, n));
    }
    double[][] reshaped = new double[p][n];
    int index = 0;
    for (int i = 0; i < p; ++i) {
      for (int j = 0; j < n; ++j) {
        reshaped[i][j] = v[index++];
      }
    }
    return reshaped;
  }

  public static double[] flat(double[][] input) {
    int dim = input[0].length;
    double[] flattened = new double[dim * input.length];
    for (int i = 0; i < input.length; ++i) {
      System.arraycopy(input[i], 0, flattened, i * dim, dim);
    }
    return flattened;
  }

  public void freeze() {
    this.freeze = true;
  }

  public double[][] getAttention() { return this.attention; }

  public void setAttention(double[][] attention) {
    this.attention = attention;
  }

  public void setId(int id) { this.id = id; }

  public void setSteps(int steps) { this.steps = steps; }

  @Override
  public int getInputDimension() {
    return this.n * this.din;
  }

  @Override
  public int getOutputDimension() {
    return this.inner.getOutputDimension();
  }

  public static double[][] matrixMult(double[][] a, double[][] b) {
    if (a[0].length != b.length) {
      throw new RuntimeException(String.format("Cannot multiply matrix of dim (%d,%d) with matrix of dim (%d,%d)", a.length, a[0].length, b.length, b[0].length));
    }
    double[][] c = new double[a.length][b[0].length];
    return matrixMult(a, b, c);
  }

  public static double[][] matrixMult(double[][] a, double[][] b, double[][] c) {
    for (int i = 0; i < a.length; ++i) {
      for (int j = 0; j < b[0].length; ++j) {
        double sum = 0.0;
        for (int k = 0; k < b.length; ++k) {
          sum += a[i][k] * b[k][j];
        }
        c[i][j] = sum;
      }
    }
    return c;
  }

  public static double[][] linearTransform(double[][] x, double[][] a, double[] b) {
    double[][] y = new double[x.length][a[0].length];
    return linearTransform(x, a, b, y);
  }

  public static double[][] linearTransform(double[][] x, double[][] a, double[] b, double[][] y) {
    matrixMult(x, a, y);
    for (int i = 0; i < y.length; ++i) {
      for (int j = 0; j < y[i].length; ++j) {
        y[i][j] += b[j];
      }
    }
    return y;
  }

  public static double[][] matrixTranspose(double[][] input) {
    double[][] act = new double[input[0].length][input.length];
    for (int i = 0; i < input.length; ++i) {
      for (int j = 0; j < input[0].length; ++j) {
        act[j][i] = input[i][j];
      }
    }
    return act;
  }

  public static double[] softmax(double[] v) {
    double sum = 0.0;
    double temp;
    for (int i = 0; i < v.length; ++i) {
      temp = Math.exp(v[i]);
      v[i] = temp;
      sum += temp;
    }
    vectorDiv(v, sum);
    return v;
  }

  public static void tanh(double[] v) {
    for (int i = 0; i < v.length; ++i) {
      v[i] = Math.tanh(v[i]);
    }
  }

  public static double[][] matrixDiv(double[][] m, double value) {
    for (double[] row : m) {
      vectorDiv(row, value);
    }
    return m;
  }

  public static void vectorDiv(double[] v, double value) {
    for (int j = 0; j < v.length; ++j) {
      v[j] /= value;
    }
  }

  @Override
  public Snapshot getSnapshot() {
    double[][][] weights = new double[1][][];
    weights[0] = reshapeVector(this.getAttentionParams(), 1, this.countParams());
    return new Snapshot(new MLPState(this.attention, weights, Domain.of(-1d, 1d)), this.getClass());
  }

}
