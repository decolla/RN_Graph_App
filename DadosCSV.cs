using Microsoft.ML.Data;

// dados lidos do CSV
public class DadosCSV
{
    [LoadColumn(0)] public string Date { get; set; }
    [LoadColumn(1)] public float PEHIST { get; set; }
    [LoadColumn(2)] public float PSHIST { get; set; }
    [LoadColumn(3)] public float REGULADOR1 { get; set; }
    [LoadColumn(4)] public float REGULADOR2 { get; set; }
    [LoadColumn(5)] public float PDT1 { get; set; }
    [LoadColumn(6)] public float PDT2 { get; set; }
    [LoadColumn(7)] public float FT1 { get; set; }
    [LoadColumn(8)] public float PEHIST_smooth_delta1 { get; set; }
    [LoadColumn(9)] public float PEHIST_smooth_delta2 { get; set; }
    [LoadColumn(10)] public float PSHIST_smooth_delta1 { get; set; }
    [LoadColumn(11)] public float PSHIST_smooth_delta2 { get; set; }
}