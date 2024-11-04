using CsvHelper.Configuration;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TextTransformation
{
    public class ReviewMap : ClassMap<Review>
    {
        public ReviewMap()
        {
            Map(m => m.ReviewText).Name("ReviewText");
            Map(m => m.Rating).Name("Rating");
            Map(m => m.Sentiment).Name("Sentiment");
            /*Map(m => m.ReviewText).Name("ReviewText");

            // Use Convert to handle custom parsing for the Rating field
            Map(m => m.Rating).Name("Rating").Convert(row =>
            {
                var ratingText = row.Row.GetField<string>("Rating");
                if (float.TryParse(ratingText, NumberStyles.Any, CultureInfo.InvariantCulture, out var rating))
                {
                    return rating;
                }
                else if (ratingText == "Like")  // Handle specific non-numeric values
                {
                    return 3.0f; // Set default value for "Like"
                }
                return null; // Return null for unrecognized text
            });


            Map(m => m.Sentiment).Name("Sentiment");*/
        }
    }
}
