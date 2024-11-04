using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TextTransformation
{
    public class LabelData
    {
        public static string ConvertRatingToSentiment(float rating)
        {
            //if (rating < 2) return "Very Negative";
            //if (rating >= 2 && rating < 3) return "Negative";
            //if (rating >= 3 && rating < 4) return "Neutral";
            //if (rating >= 4) return "Positive";
            if (rating < 3) return "Negative";
            if (rating >= 3) return "Positive";
            return "Unknown";
        }
    }
}
