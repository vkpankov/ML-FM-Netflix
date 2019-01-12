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
    class NetflixDataSetReader
    {

        [Serializable]
        public struct Rate
        {
            public short MovieId { get; set; }
            public byte Grade { get; set; }
            //public short Date { get; set; } //Days, from 1 January 2006 

        }


        public Dictionary<int, List<Rate>> ReadDataSet(string fileName)
        {
            Dictionary<int, List<Rate>> userRates = new Dictionary<int, List<Rate>>();
            List<string> lines = File.ReadAllLines(fileName).ToList();
            short currentMovieId = -1;
            DateTime maxDate = new DateTime(2006, 1, 1);
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
                    //DateTime date = DateTime.ParseExact(spLine[2], "yyyy-MM-dd", null);

                    if (!userRates.ContainsKey(userId))
                        userRates[userId] = new List<Rate>();

                    userRates[userId].Add(new Rate
                    {
                        //Date = (short)(maxDate - date).Days,
                        Grade = grade,
                        MovieId = currentMovieId
                    });
                }
            }

            return userRates;
        }


        public Matrix<double> BuildFeatureMatrixLite(Dictionary<int, List<Rate>> userRates, int movieCount, int chunkSize, int chunkId)
        {
            int userCount = userRates.Count;
            int dataCount = userRates.Skip(chunkId * chunkSize).Take(chunkSize).Sum(x => x.Value.Count);


            Matrix<double> data = Matrix<double>.Build.Sparse(dataCount, userCount + movieCount + 1);
            int ind = 0;

            for (int i = chunkId * chunkSize; i < chunkSize && i < userCount; i++)
            {
                List<Rate> uRates = userRates.Values.ElementAt(i);
                for (int j = 0; j < uRates.Count; j++)
                {
                    var uRate = uRates[j];
                    data[ind, i] = 1;
                    data[ind, userCount + uRate.MovieId - 1] = 1;
                    data[ind, data.ColumnCount - 1] = uRate.Grade;
                    ind++;
                }
            }
            return data;
        }

        /*public Matrix<int> BuildFeatureMatrix(Dictionary<int, List<Rate>> userRates, int movieCount, int takeUserCount)
        {
            int userCount = userRates.Count;
            int dataCount = userRates.Take(takeUserCount).Sum(x => x.Value.Count);
            Matrix<int> data = Matrix<int>.Build.Dense(dataCount, userCount + 3 * movieCount + 2);
            int ind = 0;
            for (int i = 0; i < userCount && ind < dataCount; i++)
            {
                List<Rate> uRates = userRates.Values.ElementAt(i);
                int lastMovieRated = -1;
                for (int j = 0; j < uRates.Count && ind < dataCount; j++)
                {
                    var uRate = uRates[j];
                    data[ind, i] = 1;
                    data[ind, userCount + uRate.MovieId - 1] = 1;
                    //!!OPTIMIZE
                    for (int k = 0; k < movieCount; k++)
                        data[ind, userCount + movieCount + k] =
                            (uRates.Exists(x => x.MovieId == k + 1) ? 1.0 : 0.0) / uRates.Count;
                    data[ind, userCount + 2 * movieCount] = uRate.Date;
                    if (lastMovieRated >= 0)
                        data[ind, userCount + 2 * movieCount + 1 + lastMovieRated] = 1;
                    lastMovieRated = uRate.MovieId - 1;
                    data[ind, data.ColumnCount - 1] = uRate.Grade;
                    ind++;
                }
            }
            return data;
        }*/
    }
}
