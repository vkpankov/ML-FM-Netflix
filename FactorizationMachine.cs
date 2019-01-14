using MathNet.Numerics.LinearAlgebra;
using MathNet.Numerics.LinearAlgebra.Double;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_FM_Netflix
{
    class FactorizationMachine
    {
        Matrix<double> V;
        Vector<double> w;
        double w0;
        double yMin, yMax;
        Random rnd = new Random();

        public FactorizationMachine(int n, int k, double targetMin, double targetMax,  double stddev = 1)
        {
            yMin = targetMin;
            yMax = targetMax;

            w0 = 0;
            w = Vector<double>.Build.Dense(n);
            V = Matrix<double>.Build.Random(n, k,
                      MathNet.Numerics.Distributions.Normal.WithMeanStdDev(0, stddev));
        }

        public double Predict(Vector<double> x, out Vector<double> vxSums)
        {
            Debug.Assert(w.Count == x.Count);
            var vxSumsBuf = Vector<double>.Build.Dense(V.ColumnCount);
            double predicted = w0 + x.DotProduct(w);

            for (int f = 0; f < V.ColumnCount; f++)
            {
                double vv = 0;
                foreach (var i in x.EnumerateIndexed(Zeros.AllowSkip))
                {
                    var val = V[i.Item1, f] * i.Item2;
                    vxSumsBuf[f] += val;
                    vv += val * val;
                }
                predicted += 1 / 2.0 * (Math.Pow(vxSumsBuf[f], 2) - vv);
            }
            vxSums = vxSumsBuf;

            return Math.Max(yMin, Math.Min(yMax, predicted));
        }

        public Vector<double> Predict(Matrix<double> x)
        {
            Vector<double> predicted = Vector<double>.Build.Dense(x.RowCount);
            for (int i = 0; i < x.RowCount; i++)
            {
                Vector<double> vSum;
                predicted[i] = this.Predict(x.Row(i), out vSum);
            }
            return predicted;
        }


        public void GradDescent(Vector<double> x, double ty, double learningRate)
        {
            int n = x.Count;
            Vector<double> vxSums;
            double ie = -2 * (ty - Predict(x, out vxSums));
            w0 = w0 - learningRate * ie;
            foreach (var i in x.EnumerateIndexed(Zeros.AllowSkip))
            {
                w[i.Item1] = w[i.Item1] - learningRate * (i.Item2 * ie);
            }
            for (int f = 0; f < V.ColumnCount; f++)
            {
                foreach (var i in x.EnumerateIndexed(Zeros.AllowSkip))
                {
                    V[i.Item1, f] = V[i.Item1, f] - learningRate *
                        ((i.Item2 * vxSums[f] - V[i.Item1, f] * Math.Pow(i.Item2, 2)) * ie);
                }
            }
        }

        public double Learn(List<Chunk> trainData, int skipIndex, double learningRate, int itCount, double err = 1.08)
        {
            double rmse = 5;
            
            for (int it = 0; it < itCount; it++)
            {
                int startTime = Environment.TickCount;
                NetflixDataHelper.Shuffle(ref trainData);
                for (int k = 0; k<trainData.Count; k++)
                {
                    if (k == skipIndex)
                        continue;
                    Chunk chunk = trainData[k];
                    int n = chunk.X.RowCount;
                    while (n > 1)
                    {
                        n--;
                        int m = rnd.Next(n + 1);
                        this.GradDescent(chunk.X.Row(m), chunk.Y[m], learningRate);
                    }
                }
                Vector<double> testEval = this.Predict(trainData[skipIndex].X);
                Vector<double> e = testEval - trainData[skipIndex].Y;
                rmse = Math.Sqrt(e.PointwisePower(2).Sum() / e.Count);
                var time = (Environment.TickCount - startTime) / 1000.0;
                Console.WriteLine($"Test batch - {skipIndex}, iteration - {it}, rmse: {rmse}, time: {time}");
                if (rmse < err)
                    break;
            }
            return rmse;
        }
    }
}
