using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using MathNet.Numerics;
using MathNet.Numerics.LinearAlgebra;
using System.Runtime.Serialization.Formatters.Binary;
using MathNet.Numerics.LinearAlgebra.Double;

namespace ML_FM_Netflix
{
    public static class NetflixDataHelper
    {
        public static void Shuffle<T>(ref List<T> data)
        {
            Random rng = new Random();
            int n = data.Count;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var value = data[k];
                data[k] = data[n];
                data[n] = value;
            }
        }
        public static void Shuffle(Chunk chunk)
        {
            Random rng = new Random();
            int n = chunk.X.RowCount;
            while (n > 1)
            {
                n--;
                int k = rng.Next(n + 1);
                var valueX = chunk.X.Row(k);
                var valueY = chunk.Y[k];
                chunk.X.SetRow(k, chunk.X.Row(n));
                chunk.Y[k] = chunk.Y[n];
                chunk.X.SetRow(n, valueX);
                chunk.Y[n] = valueY;
            }
        }


        public static List<Rate> ReadDataSet(string fileName, int totalGradeCount = 24053764, int userCount = 470758, int movieCount = 4500)
        {
            List<Rate> allRates = new List<Rate>();
            List<string> lines = File.ReadAllLines(fileName).ToList();
            StringBuilder liteFile = new StringBuilder();
            Dictionary<int, int> userIds = new Dictionary<int, int>();

            short currentMovieId = -1;
            int currentUserId = 0;

            foreach (var line in lines)
            {
                if (line.Contains(":"))
                {
                    currentMovieId = Int16.Parse(line.TrimEnd(':'));
                }
                else
                {
                    string[] spLine = line.Split(',');
                    int userId = Int32.Parse(spLine[0]);
                    byte grade = Byte.Parse(spLine[1]);

                    int matrixUserId = 0;
                    if (!userIds.ContainsKey(userId))
                    {
                        userIds.Add(userId, currentUserId++);
                        matrixUserId = currentUserId - 1;
                    }
                    else
                        matrixUserId = userIds[userId];

                    allRates.Add(new Rate { UserId = matrixUserId, Grade = grade, MovieId = currentMovieId });
                }
            }
            return allRates;
        }

        public static  List<Chunk> GetChunks(List<Rate> userRates, int userCount, int movieCount, int chunkCount)
        {
            List<Chunk> chunks = new List<Chunk>();
            int chunkSize = userRates.Count / chunkCount;
            int j = 0;
            for (int i = 0; i < chunkCount; i++)
            {
                Matrix<double> x = Matrix<double>.Build.Sparse(chunkSize, userCount + movieCount);
                Vector<double> y = Vector<double>.Build.Sparse(chunkSize);
                for (int k = 0; k < chunkSize && j<userRates.Count; k++)
                {
                    x[k, userRates[j].UserId] = 1;
                    x[k, userCount + userRates[j].MovieId - 1] = 1;
                    y[k] = userRates[j].Grade;
                    j++;
                }
                chunks.Add(new Chunk { X = x, Y = y });
            }
            return chunks;
        }

    }
}
