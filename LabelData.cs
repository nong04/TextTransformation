namespace TextTransformation
{
    public class LabelData
    {
        public static string ConvertRatingToSentiment(float rating)
        {
            if (rating < 3) return "Negative";
            if (rating >= 3 && rating <= 4) return "Neutral";
            if (rating > 4) return "Positive";
            return "Unknown";
        }
    }
}
