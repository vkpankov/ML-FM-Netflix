using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics.Statistics;

namespace ML_FM_Netflix
{
    class Program
    {
        static Chunk JoinChunks(List<Chunk> chunks)
        {
            Chunk resChunks = new Chunk();
            int totalRowCount = chunks.Select(m => m.X.RowCount).Sum();
            resChunks.X = Matrix<double>.Build.Sparse(totalRowCount, chunks[0].X.ColumnCount);
            resChunks.Y = Vector<double>.Build.Sparse(totalRowCount);
            for (int i = 0; i < chunks.Count; i++)
            {
                int curCount = chunks[i].X.RowCount;
                resChunks.X.SetSubMatrix(i * curCount, 0, chunks[i].X);
                resChunks.Y.SetSubVector(i * curCount, curCount, chunks[i].Y);
            }
            return resChunks;
        }

        static void Main(string[] args)
        {
            const int usersCount = 470758, moviesCount = 4500;
            const double targetMin = 1, targetMax = 5;
            const int VSize = 3;
            const double learningRate = 0.001;
            const double maxError = 1.08;
            const int maxIterations = 4;
            const int batchCount = 15500;

            List<Rate> rates = NetflixDataHelper.ReadDataSet( @"combined_data_1.txt");
            Console.WriteLine("Dataset read: " + rates.Count + " records");

            List<Chunk> chunks = NetflixDataHelper.GetChunks(rates, usersCount, moviesCount, batchCount);
            Console.WriteLine("Feature matrix created");

            Vector<double> rmseChunks = Vector<double>.Build.Dense(chunks.Count);

            for (int k = 0; k < chunks.Count; k++)
            {
                int startTime = Environment.TickCount;
                NetflixDataHelper.Shuffle(ref chunks);
                FactorizationMachine fm =
                    new FactorizationMachine(chunks[0].X.ColumnCount, VSize, targetMin, targetMax, 0.1);
                double kRMSE = fm.Learn(chunks, k, learningRate, maxIterations, maxError);
                rmseChunks[k] = kRMSE;
            }
            var rmseMean = rmseChunks.Mean();
            var rmseStdDev = rmseChunks.StandardDeviation();
            Console.WriteLine("RMSE mean: " + rmseMean + "+-" + rmseStdDev);
            Console.ReadKey();
        }
    }
}
