using System;
using System.Xml.Serialization;

namespace ML_FM_Netflix
{
    public struct Rate
    {
        public int UserId { get; set; }
        public short MovieId { get; set; }
        public byte Grade { get; set; }
        //public short Date { get; set; } //Days, from 1 January 2006 

    }

}
