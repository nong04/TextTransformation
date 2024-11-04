using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TextTransformation
{
    public class ReviewPrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedSentiment { get; set; }
        public float[] Score { get; set; }
    }
}
