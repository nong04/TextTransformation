//Source: https://devblogs.microsoft.com/dotnet/introducing-the-ml-dotnet-text-classification-api-preview/
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.Data.Analysis;
using Microsoft.ML.TorchSharp;
using System;
using System.IO;
using System.Net;
using System.Dynamic;
using System.Formats.Asn1;
using System.Globalization;
using System.Text;
using TextTransformation;
using TextProcedure;
using CsvHelper;
using CsvHelper.Configuration;
using System.Text.RegularExpressions;

class Program
{
    static void Main(string[] args)
    {
        string savePath1 = @"..\..\Restaurant reviews.csv"; // Original file (after removing useless columns and some shit)
        string savePath2 = @"..\..\dataset2.csv"; // after removing duplicate data
        string savePath3 = @"..\..\dataset3.csv"; // after expanding acronyms
        string savePath4 = @"..\..\dataset4.csv"; // after removing punctuations
        string savePath5 = @"..\..\dataset5.csv"; // after lowering case text
        string savePath6 = @"..\..\dataset6.csv"; // after assigning labels

        string savePaths1 = @"..\..\smalldataset1.csv";
        string savePaths2 = @"..\..\smalldataset2.csv";

        List<Review> data = ReadCSV(savePath6);

        //PrintData(data);

        //CheckDuplicateData(data);
        //List<Review> data2 = RemoveDuplicateData(data);
        //SaveReviewsToCsv(data2, @"..\..\dataset2.csv");

        //List<Review> data3 = ExpandCharacters(data2);
        //SaveReviewsToCsv(data3, @"..\..\dataset3.csv");

        //List<Review> data4 = RemovePunctuation(data3);
        //SaveReviewsToCsv(data4, @"..\..\dataset4.csv");

        //List<Review> data5 = ToLower(data4);
        //SaveReviewsToCsv(data5, @"..\..\dataset5.csv");

        //List<Review> data6 = AssignLabel(data5);
        //SaveReviewsToCsv(data6, @"..\..\dataset6.csv");

        //List<Review> data2 = AssignLabel(data);
        //SaveReviewsToCsv(data2, @"..\..\smalldataset2.csv");

        TrainingData(data);
        //Test();
        //Console.WriteLine("Done");
    }

    public static List<Review> ReadCSV(string filePath)
    {
        using (var fileStream = new FileStream(filePath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite))
        using (var reader = new StreamReader(fileStream))
        using (var csv = new CsvReader(reader, new CsvConfiguration(CultureInfo.InvariantCulture)
        {
            HasHeaderRecord = true,
            MissingFieldFound = null, // Ignore missing fields if any
            HeaderValidated = null, // Ignore header validation
            BadDataFound = null // Ignore bad data
        }))
        {
            csv.Context.RegisterClassMap<ReviewMap>();

            // Load records, handle empty Sentiment
            var records = csv.GetRecords<Review>().Select(r => new Review
            {
                ReviewText = r.ReviewText,
                Rating = r.Rating,
                Sentiment = string.IsNullOrEmpty(r.Sentiment) ? "Unknown" : r.Sentiment // Default to "Unknown" if empty
            }).ToList();

            return records;
        }
    }

    public static void SaveReviewsToCsv(List<Review> reviews, string filePath)
    {
        using (var writer = new StreamWriter(filePath))
        using (var csv = new CsvWriter(writer, CultureInfo.InvariantCulture))
        {
            csv.WriteRecords(reviews); // Write the list to the CSV file
        }
        Console.WriteLine($"Data saved to: {filePath}");
    }

    private static void PrintData(List<Review> data)
    {
        foreach (Review obj in data)
        {
            Console.WriteLine(obj.ReviewText);
            Console.WriteLine(obj.Rating);
            Console.WriteLine(obj.Sentiment);
            Console.WriteLine("-------------------------");
        }
    }

    private static void CheckDuplicateData(List<Review> data)
    {
        var duplicateRecords = data
                .Select((review, index) => new { review, index })
                .GroupBy(record => new
                {
                    record.review.ReviewText,  // Assuming Review is a property of Review class
                    record.review.Rating   // Assuming Rating is a property of Review class
                })
                .Where(group => group.Count() > 1)
                .ToList();

        // Output duplicates
        if (duplicateRecords.Count > 0)
        {
            Console.WriteLine("Duplicate records found:");
            foreach (var group in duplicateRecords)
            {
                Console.WriteLine($"Duplicate ReviewText: \"{group.Key.ReviewText}\", Rating: {group.Key.Rating}");
                Console.WriteLine("Positions: " + string.Join(", ", group.Select(x => x.index)));
                Console.WriteLine($"Count: {group.Count()}");
                Console.WriteLine();
            }
        }
        else
        {
            Console.WriteLine("No duplicate records found.");
        }
    }

    private static List<Review> RemoveDuplicateData(List<Review> data)
    {
        var uniqueRecords = data
                .GroupBy(record => new{record.ReviewText, record.Rating})
                .Select(group => group.First())
                .ToList();
        return uniqueRecords;
    }

    public static List<Review> ExpandCharacters(List<Review> data)
    {
        List<Review> data2 = new List<Review>();

        foreach (Review rv in data)
        {
            string x = rv.ReviewText;
            x = x.Replace("\u2019", "'");
            string y = TextExpander.ExpandAcronyms(x);

            Review review = new Review(y, rv.Rating);
            data2.Add(review);
        }
        return data2;
    }

    public static List<Review> RemovePunctuation(List<Review> input)
    {
        List<Review> output = new List<Review>();
        foreach (Review review in input)
        {
            StringBuilder sb = new StringBuilder();
            string input2 = Regex.Replace(review.ReviewText, @"[^a-zA-Z0-9\s.,!?]", "");
            foreach (char c in input2)
            {
                if (!char.IsPunctuation(c))
                {
                    sb.Append(c);
                }
            }
            Review review2 = new Review(sb.ToString(), review.Rating);
            output.Add(review2);
        }
        return output;
    }

    private static List<Review> ToLower(List<Review> data)
    {
        List<Review> result = new List<Review>();
        foreach (Review rv in data)
        {
            Review rv2 = new Review(rv.ReviewText.ToLower(), rv.Rating);
            result.Add(rv2);
        }
        return result;
    }

    private static List<Review> AssignLabel(List<Review> data)
    {
        List<Review> ret = new List<Review>();
        foreach (Review review in data)
        {
            Review rv = new Review(review.ReviewText, review.Rating, LabelData.ConvertRatingToSentiment((float)review.Rating));
            ret.Add(rv);
        }
        return ret;
    }

    public static void TrainingData(List<Review> reviews)
    {
        // Initialize MLContext
        MLContext mlContext = new()
        {
            GpuDeviceId = 0,
            FallbackToCpu = true
        };

        var data = mlContext.Data.LoadFromEnumerable(reviews);

        //foreach (var row in df.Rows)
        //{
        //    Console.WriteLine(string.Join("\t", row));
        //}

        // Split the data into train and test sets.
        var trainTestSplit = mlContext.Data.TrainTestSplit(data, testFraction: 0.2);
        var trainData = trainTestSplit.TrainSet;
        var testData = trainTestSplit.TestSet;

        //Define your training pipeline
        var pipeline = mlContext.Transforms.Text.NormalizeText(outputColumnName: "ReviewText", inputColumnName: nameof(Review.ReviewText),
                keepDiacritics: false, keepPunctuations: false, keepNumbers: true)
            .Append(mlContext.Transforms.Conversion.MapValueToKey(outputColumnName: "Label", nameof(Review.Sentiment)))
            .Append(mlContext.MulticlassClassification.Trainers.TextClassification(sentence1ColumnName: "ReviewText"))
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        Console.WriteLine("Defined train sets");

        // Train the model
        var model = pipeline.Fit(trainData);
        Console.WriteLine("Trained the model");

        // Use the model to make predictions
        var predictionIDV = model.Transform(testData);
        Console.WriteLine("Tested the model");

        // The result of calling Transform is an IDataView with your predicted values. To make it easier to view your predictions, convert the IDataView to an IDataFrame.
        //var columnsToSelect = new[] { "ReviewText", "Sentiment", "PredictedLabel" };
        //var predictions = predictionIDV.ToDataFrame(columnsToSelect);
        //Console.WriteLine("Text\tSentiment\tPredictedLabel");
        //foreach (var row in predictions.Rows)
        //{
        //    Console.WriteLine(string.Join("\t", row));
        //}

        // Evaluate the model
        var evaluationMetrics = mlContext.MulticlassClassification.Evaluate(predictionIDV);

        Console.WriteLine($"MacroAccuracy: {evaluationMetrics.MacroAccuracy:F4}");
        Console.WriteLine($"MicroAccuracy: {evaluationMetrics.MicroAccuracy:F4}");
        Console.WriteLine($"LogLoss: {evaluationMetrics.LogLoss:F4}");
        //Console.WriteLine($"LogLossReduction: {evaluationMetrics.LogLossReduction:F2}");
        //for (int i = 0; i < evaluationMetrics.PerClassLogLoss.Count; i++)
        //{
        //    Console.WriteLine($"LogLoss in class {i}: {evaluationMetrics.PerClassLogLoss[i]:F2}");
        //}

        Console.WriteLine(evaluationMetrics.ConfusionMatrix.GetFormattedConfusionTable());

        // Save the model
        string modelPath = $"D:/Documents/_Programming_/DMML/TextTransformation/bin/TextTransformation_{evaluationMetrics.MacroAccuracy:F4}_{evaluationMetrics.MicroAccuracy:F4}.zip";
        mlContext.Model.Save(model, trainData.Schema, modelPath);
        Console.WriteLine($"Model saved to: {modelPath}");
    }

    private static string PredictFromModel(string modelPath, string review)
    {
        var mlContext = new MLContext();
        DataViewSchema modelSchema;
        var loadedModel = mlContext.Model.Load(modelPath, out modelSchema);

        // Prediction example with loaded model
        var predictionEngine = mlContext.Model.CreatePredictionEngine<Review, ReviewPrediction>(loadedModel);
        var sampleReview = new Review { ReviewText = review };
        var prediction = predictionEngine.Predict(sampleReview);

        return prediction.PredictedSentiment;
    }

    private static void Test()
    {
        var feedbackSamples = new[]
{
    "The pumpkin spice latte was delightful! I can't get enough of it.",
    "I visited Starbucks, and the service was incredibly slow. I was very disappointed.",
    "The coffee was great, but the atmosphere was a bit too loud for my liking.",
    "I love the seasonal flavors! They always bring something new and exciting.",
    "My experience was average; the coffee was fine, but nothing extraordinary.",
    "The staff was friendly, but my order was wrong. I had to wait again for it to be corrected.",
    "I enjoy sitting in the cozy corner with a book and a hot drink. A perfect afternoon!",
    "The new holiday drinks are fantastic! I can’t wait to try them all.",
    "Unfortunately, my last visit was a letdown. The mocha was too sweet and left a bad aftertaste.",
    "I had a wonderful time chatting with friends over coffee. The vibe is always welcoming.",
    "The location is great, but parking is a hassle. It often discourages me from visiting.",
    "The iced coffee was refreshing, especially on a hot day. I'll be back for more!",
    "I don't usually drink coffee, but the teas here are superb! I highly recommend them.",
    "I waited too long in line, and when I got my drink, it was lukewarm.",
    "Starbucks is my go-to place for meetings. It's convenient and the Wi-Fi is reliable.",
    "I've always had good experiences here. The baristas know their stuff!",
    "I tried the new oat milk option, and it was delicious! A great alternative.",
    "I was disappointed with the cleanliness of the shop. It could use some attention.",
    "The loyalty program is fantastic! I love getting rewards for my purchases.",
    "Every time I go, I find something new to love. Highly recommend!",
    "i ordered a grande tea and they used only one tea bag, the same as a tall tea. what is the extra charge for in the grande size? water? come on, give me a break.",
    "i order the same things at many starbucks in california. this is the only starbucks that charges me more for the same product. i always order a grande americano with steamed breve. i am charged 65 cents more. i am told it is everything from the breve to the labor. everywhere else it is $2.55."
};
        string modelPath = @"..\..\SentimentModel_0.7982_0.8743.zip";

        for (int i = 0; i < feedbackSamples.Length; i++)
        {
            Console.WriteLine(feedbackSamples[i] + " :   " + PredictFromModel(modelPath, feedbackSamples[i]));
        }
    }
}