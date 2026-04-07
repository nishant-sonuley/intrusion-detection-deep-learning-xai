using Microsoft.ML;
using Microsoft.ML.Data;
using System.Diagnostics;
using System.Collections.Generic;

var mlContext = new MLContext();

string modelPath = "cnn_model_phase7_single.onnx";

var pipeline = mlContext.Transforms.ApplyOnnxModel(
    modelFile: modelPath,
    outputColumnNames: new[] { "output" },
    inputColumnNames: new[] { "input" });

var emptyData = mlContext.Data.LoadFromEnumerable(new List<ModelInput>());

var model = pipeline.Fit(emptyData);

var predictor = mlContext.Model.CreatePredictionEngine<ModelInput, ModelOutput>(model);

Console.WriteLine("Model loaded successfully.");

// Sample input
var random = new Random();
float[] sample = new float[122];

for (int i = 0; i < 122; i++)
{
    sample[i] = (float)random.NextDouble();
}

var input = new ModelInput { input = sample };

// Benchmark
int iterations = 10000;
var stopwatch = Stopwatch.StartNew();

for (int i = 0; i < iterations; i++)
{
    predictor.Predict(input);
}

stopwatch.Stop();

double avgTime = stopwatch.Elapsed.TotalMilliseconds / iterations;

Console.WriteLine($"Average inference time per sample: {avgTime} ms");

public class ModelInput
{
    [VectorType(1, 122)]
    public float[] input { get; set; } = default!;
}

public class ModelOutput
{
    [VectorType(4)]
    public float[] output { get; set; } = default!;
}
