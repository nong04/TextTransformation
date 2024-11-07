using System.Text.RegularExpressions;

namespace TextProcedure
{
    public class TextExpander
    {
        public static string ExpandAcronyms(string input)
        {
            string pattern = @"\b(?i)('m|'re|'s|'d|'ll|'ve|n't|can't|won't|isn't|wasn't|aren't|don't|doesn't|haven't|hadn't|didn't|couldn't)\b";

            input = Regex.Replace(input, pattern, match =>
            {
                switch (match.Value.ToLower())
                {
                    case "'m": return " am";
                    case "'re": return " are";
                    case "'s": return " is";
                    case "'d": return " would";
                    case "'ll": return " will";
                    case "'ve": return " have";
                    case "wouldn't": return "would not";
                    case "shouldn't": return "should not";
                    case "can't": return "can not";
                    case "won't": return "will not";
                    case "isn't": return "is not";
                    case "wasn't": return "was not";
                    case "aren't": return "are not";
                    case "don't": return "do not";
                    case "doesn't": return "does not";
                    case "haven't": return "have not";
                    case "hadn't": return "had not";
                    case "didn't": return "did not";
                    case "couldn't": return "could not";

                    default: return match.Value;
                }
            });
            return input.Trim();
        }
    }
}
