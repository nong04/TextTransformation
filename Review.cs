using Microsoft.ML.Data;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace TextTransformation
{
    public class Review
    {
        [LoadColumn(0)]  // Replace 0 with the actual column index for ReviewText in your CSV file
        public string ReviewText { get; set; }

        [LoadColumn(1)]  // Replace 1 with the actual column index for Rating in your CSV file
        public float Rating { get; set; }

        [LoadColumn(2)]  // Replace 2 with the actual column index for Sentiment in your CSV file
        public string Sentiment { get; set; }
        public Review() { }

        public Review(string ReviewText)
        {
            this.ReviewText = ReviewText;
        }


        public Review(string ReviewText, float Rating) 
        {
            this.ReviewText = ReviewText;
            this.Rating = Rating;
        }

        public Review(string reviewText, float rating, string sentiment) 
        {
            this.ReviewText = reviewText;
            this.Rating = rating;
            this.Sentiment = sentiment;
        }

        
    }
}
