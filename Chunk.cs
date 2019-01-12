using MathNet.Numerics.LinearAlgebra;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace ML_FM_Netflix
{
    public struct Chunk
    {
        public Matrix<double> X { get; set; }
        public Vector<double> Y { get; set; }
    }
}
