import it.units.erallab.hmsrobots.core.controllers.MultiLayerPerceptron;
import org.ejml.simple.SimpleMatrix;
import org.junit.Test;
import world.units.erallab.SelfAttention;

import java.util.Random;

import static org.junit.Assert.*;


public class testSelfAttention {

  private static SimpleMatrix getRandomMatrix(int dim1, int dim2) {
    return SimpleMatrix.random(dim1, dim2, 0, 100, new Random(0));
  }

  private static double[] getRandomVector(int dim) {
    Random random = new Random(0);
    double[] vec = new double[dim];
    for (int i = 0; i < dim; ++i) {
      vec[i] = random.nextDouble();
    }
    return vec;
  }

  private static double[][] matrixToBiArray(SimpleMatrix matrix) {
    double[][] array = new double[matrix.numRows()][matrix.numCols()];
    for (int r = 0; r < matrix.numRows(); r++) {
      for (int c = 0; c < matrix.numCols(); c++) {
        array[r][c] = matrix.get(r, c);
      }
    }
    return array;
  }

  private static double[] matrixToArray(SimpleMatrix matrix) {
    double[] array = new double[matrix.numRows() * matrix.numCols()];
    int i = 0;
    for (int r = 0; r < matrix.numRows(); r++) {
      for (int c = 0; c < matrix.numCols(); c++) {
        array[i++] = matrix.get(r, c);
      }
    }
    return array;
  }

  private static SelfAttention getTestInstance() {
    return new SelfAttention(new MultiLayerPerceptron(MultiLayerPerceptron.ActivationFunction.TANH, 10, new int[]{}, 1), 5, 5, 2, 2);
  }

  @Test
  public void testConcat() {
    double[] c = new double[] {1, 2, 3, 4};
    double[] b = new double[] {5, 6, 7, 8};
    double[] a = new double[] {9, 10, 11, 12};
    assertArrayEquals(new double[] {9, 10, 11, 12, 5, 6, 7, 8, 1, 2, 3, 4}, SelfAttention.concat(a, b, c), 0.00000001);
  }

  @Test
  public void testReshape() {
    SimpleMatrix a = getRandomMatrix(10, 1);
    a.reshape(5, 2);
    assertArrayEquals(matrixToBiArray(a), SelfAttention.reshapeVector(matrixToArray(a), 5, 2));
  }

  @Test
  public void testVectorMult() {
    SimpleMatrix a = getRandomMatrix(5, 1);
    SimpleMatrix b = getRandomMatrix(1, 5);
    assertArrayEquals(matrixToBiArray(a.mult(b)), SelfAttention.matrixMult(matrixToBiArray(a), matrixToBiArray(b)));
  }

  @Test
  public void testMatrixMult() {
    SimpleMatrix a = getRandomMatrix(10, 10);
    SimpleMatrix b = getRandomMatrix(10, 10);
    assertArrayEquals(matrixToBiArray(a.mult(b)), SelfAttention.matrixMult(matrixToBiArray(a), matrixToBiArray(b)));
  }

  @Test
  public void testMatrixMultBis() {
    SimpleMatrix a = getRandomMatrix(10, 4);
    SimpleMatrix b = getRandomMatrix(4, 10);
    assertArrayEquals(matrixToBiArray(a.mult(b)), SelfAttention.matrixMult(matrixToBiArray(a), matrixToBiArray(b)));
  }

  @Test
  public void testLinearTransform() {
    SimpleMatrix a = getRandomMatrix(10, 4);
    SimpleMatrix b = getRandomMatrix(4, 10);
    SimpleMatrix c = getRandomMatrix(1, 10);
    SimpleMatrix d = c.copy();
    for (int i = 0; i < c.getNumElements(); ++i) {
      d = d.combine(i, 0, c.copy());
    }
    assertArrayEquals(matrixToBiArray(a.mult(b).plus(d)), SelfAttention.linearTransform(matrixToBiArray(a), matrixToBiArray(b), matrixToArray(c)));
  }

  @Test
  public void testTranspose() {
    SimpleMatrix a = getRandomMatrix(5, 5);
    assertArrayEquals(matrixToBiArray(a.transpose()), SelfAttention.matrixTranspose(matrixToBiArray(a)));
  }

  @Test
  public void testMatrixDiv() {
    SimpleMatrix a = getRandomMatrix(5, 5);
    double value = 5.0;
    assertArrayEquals(matrixToBiArray(a.divide(value)), SelfAttention.matrixDiv(matrixToBiArray(a), value));
  }

  @Test
  public void testBasicExecution() {
    double[] v = getRandomVector(5);
    assertEquals(1, getTestInstance().apply(v).length);
    double[][] attention = getTestInstance().applyAttention(v);
    assertEquals(5, attention.length);
    for (double[] doubles : attention) {
      assertEquals(2, doubles.length);
    }
  }

  @Test
  public void testParams() {
    SelfAttention test = getTestInstance();
    assertEquals(36, test.getAttentionParams().length);
  }

}
