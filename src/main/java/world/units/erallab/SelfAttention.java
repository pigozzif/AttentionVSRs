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

  private final double[][] attention;
  private double[][] latentCode;
  private final double[][] q;
  private final double[][] k;
  private final double[][] v;

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
    this.attention = new double[n][n];
    this.latentCode = new double[n][dv];
    this.q = new double[n][dk];
    this.k = new double[n][dk];
    this.v = new double[n][dv];
  }

  public SelfAttention(MultiLayerPerceptron inner, int n, int din, int dk, int dv) {
    this(inner, n, din, dk, dv, new double[din][dk], new double[din][dk], new double[din][dv],
            new double[dk], new double[dk], new double[dv]);
  }

  public static double[][] oneHot(int n, int id) {
    double[][] vec = new double[n][1];
    vec[id][0] = 1.0;
    return vec;
  }
  // TODO: rename
  public static int countParams(int din, int dk, int dv) {
    return (din * dk) + dk + (din * dk) + dk + (din * dv) + dv;
  }

  public int countParams() { return countParams(this.din, this.dk, this.dv); }

  public double[] getAttentionParams() { return concat(flat(this.wq), flat(this.wk), flat(this.wv), this.qbias, this.kbias, this.vbias); }

  public double[] getDownstreamParams() { return this.inner.getParams(); }

  @Override
  public double[] getParams() { return concat(this.getAttentionParams(), this.getDownstreamParams()); }

  public void setAttentionParams(double[] params) {
    int s = 0;
    for (int i = 0; i < this.din; ++i) {
      System.arraycopy(params, s, this.wq[i], 0, this.dk);
      s = s + this.dk;
    }
    System.arraycopy(params, s, this.qbias, 0, this.dk);
    s = s + this.dk;
    for (int i = 0; i < this.din; ++i) {
      System.arraycopy(params, s, this.wk[i], 0, this.dk);
      s = s + this.dk;
    }
    System.arraycopy(params, s, this.kbias, 0, this.dk);
    s = s + this.dk;
    for (int i = 0; i < this.din; ++i) {
      System.arraycopy(params, s, this.wv[i], 0, this.dv);
      s = s + this.dv;
    }
    System.arraycopy(params, s, this.vbias, 0, this.dv);
  }

  public void setDownstreamParams(double[] params) { this.inner.setParams(params); }

  @Override
  public void setParams(double[] params) {
    this.setAttentionParams(Arrays.stream(params).limit(countParams(this.din, this.dk, this.dv)).toArray());
    this.setDownstreamParams(Arrays.stream(params).skip(countParams(this.din, this.dk, this.dv)).toArray());
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
    return this.inner.apply(flat(this.applyAttention(inputs)));
  }

  public double[][] applyAttention(double[] inputs) {
    double[][] reshaped = reshapeVector(inputs, this.n, this.din);
    linearTransform(reshaped, this.wq, this.qbias, this.q);
    double[][] keys = matrixTranspose(linearTransform(reshaped, this.wk, this.kbias, this.k));
    linearTransform(reshaped, this.wv, this.vbias, this.v);
    matrixMult(this.q, keys, this.attention);
    matrixDiv(this.attention, Math.sqrt(this.dk));
    for (double[] row : this.attention) {
      /*tanh(row);*/softmax(row);
    }
    //this.latentCode = this.attention;
    return /*this.latentCode;*/matrixMult(this.attention, this.v, this.latentCode);
  }

  public static double[][] positionalEncoding(double[] inputs, double[][] posEmbedding) {
    return matrixMult(posEmbedding, reshapeVector(inputs, 1, inputs.length));
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

  public static double[] tanh(double[] v) {
    for (int i = 0; i < v.length; ++i) {
      v[i] = Math.tanh(v[i]);
    }
    return v;
  }

  public static double[][] matrixDiv(double[][] m, double value) {
    for (double[] row : m) {
      vectorDiv(row, value);
    }
    return m;
  }

  public static double[] vectorDiv(double[] v, double value) {
    for (int j = 0; j < v.length; ++j) {
      v[j] /= value;
    }
    return v;
  }

  @Override
  public Snapshot getSnapshot() {
    double[][][] weights = new double[1][][];
    weights[0] = reshapeVector(this.getAttentionParams(), 1, this.countParams());
    return new Snapshot(new MLPState(this.attention, weights, Domain.of(-1d, 1d)), this.getClass());
  }

}