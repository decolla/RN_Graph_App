namespace RN_Graph_App;
using System;
using System.Collections.Generic;
using System.Drawing;
using System.IO;
using System.Linq;
using System.Windows.Forms;
using Microsoft.ML;
using Microsoft.ML.Data;
using ZedGraph;

// classe dados categorizados no CSV (analisados por IA)
public class DadosSensor
{
    [LoadColumn(0)] public string Date { get; set; }
    [LoadColumn(1)] public float PEHIST { get; set; }
    [LoadColumn(2)] public float PSHIST { get; set; }
    [LoadColumn(3)] public float REGULADOR1 { get; set; }
    [LoadColumn(4)] public float REGULADOR2 { get; set; }
    [LoadColumn(5)] public float PDT1 { get; set; }
    [LoadColumn(6)] public float PDT2 { get; set; }
    //[LoadColumn(7)] public float ZASL1 { get; set; }
    //[LoadColumn(8)] public float ZASL2 { get; set; }
    [LoadColumn(9)] public float FT1 { get; set; } 
}

public class PrevisaoOutput
{
    // o "score" é padrão no ML.NET, mas para ONNX pode variar.
    [ColumnName("outputs")] 
    public float[] PredictedValue { get; set; }
}

public partial class Form1 : Form
{
    // caminho do arquivo ONNX
    public string path_ONNX = "";
    
    // lista de dados categorizados
    private List<DadosSensor> dadosCarregados = new List<DadosSensor>();
    public Form1()
    {
        InitializeComponent();
    }
    
    // botão de carregar arquivo CSV
    private void button1_Click(object sender, EventArgs e)
    {
        using (OpenFileDialog openFileDialog = new OpenFileDialog() { Filter = "CSV Files|*.csv" })
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                ReadCSV(openFileDialog.FileName);
            }
        }
    }

    // botão de carregar arquivo ONNX
    private void button2_Click(object sender, EventArgs e)
    {
        using (OpenFileDialog openFileDialog = new OpenFileDialog() { Filter = "ONNX Files|*.onnx" })
        {
            if (openFileDialog.ShowDialog() == DialogResult.OK)
            {
                path_ONNX = openFileDialog.FileName;
                MessageBox.Show("Modelo Selecionado!");
            }
        }
    }

    // botão de processar o arquivo 
    private void button3_Click(object sender, EventArgs e)
    {
        if (dadosCarregados.Count == 0 || path_ONNX == "")
        {
            MessageBox.Show("Carregue um arquivo CSV e um modelo ONNX");
            return;
        }
    
        MLContext mlContext = new MLContext();
        IDataView dataView = mlContext.Data.LoadFromEnumerable(dadosCarregados);
        Console.WriteLine(dadosCarregados.ToArray().Length);
        
        // colunas que precisam ser concatenadas
        string[] colunasEntrada = new[] { "PEHIST", "PSHIST", "REGULADOR1", "REGULADOR2", "PDT1",/* "ZASL1", "ZASL2",*/ "PDT2" };

        // nome de entrada e saída do ONNX
        string nomeEntradaONNX = "batch_x";
        string nomeEntradaONNX2 = "batch_x_mark";
        string nomeSaidaONNX = "outputs"; 
    
        // pipeline para concatenar e aplicar o modelo
        var pipeline = mlContext.Transforms.Concatenate(nomeEntradaONNX, colunasEntrada)
            .Append(mlContext.Transforms.ApplyOnnxModel(
                modelFile: path_ONNX,
                outputColumnNames: new[] { nomeSaidaONNX },
                inputColumnNames: new[] { nomeEntradaONNX },
                gpuDeviceId: null, 
                fallbackToCpu: true
            ));
            
        Console.WriteLine(pipeline.ToString());

        try
        {
            // executar
            var transformer = pipeline.Fit(dataView);
            var transformedData = transformer.Transform(dataView);

            // extração dos resultados
            var predictions = mlContext.Data.CreateEnumerable<PrevisaoOutput>(transformedData, reuseRowObject: false).ToList();
            
            // Preparar listas para o gráfico
            List<double> yReal = dadosCarregados.Select(d => (double)d.FT1).ToList();
            List<double> yPredito = predictions.Select(p => (double)p.PredictedValue[0]).ToList();

            PlotarGrafico(yReal, yPredito);
        }
        catch (Exception ex)
        {
            MessageBox.Show($"Erro durante o processamento: {ex.Message}", "Erro");
        }
    }
        
    
    // função para ler o arquivo CSV
    private void ReadCSV(string path)
    {
        dadosCarregados.Clear(); // Limpa lista antiga para não duplicar
        // le as linhas do arquivo CSV e armazena em uma lista
        var linhas = File.ReadAllLines(path);
        
        // carrega os dados
        for (int i = 1; i < linhas.Length; i++)
        {
            var cols = linhas[i].Split(','); // divide a linha em colunas
            Console.WriteLine(cols[7]);
            if (cols.Length < 7) continue; // proteção contra linhas vazias
            try
            {
                var item = new DadosSensor
                {
                    Date = cols[0],
                    PEHIST = ParseFloat(cols[1]),
                    PSHIST = ParseFloat(cols[2]),
                    REGULADOR1 = ParseFloat(cols[3]),
                    REGULADOR2 = ParseFloat(cols[4]),
                    PDT1 = ParseFloat(cols[5]),
                    PDT2 = ParseFloat(cols[6]),
                    /*
                    ZASL1 = ParseFloat(cols[7]),
                    ZASL2 = ParseFloat(cols[8]),
                    */
                    FT1 = ParseFloat(cols[7])
                };
                Console.WriteLine(item.FT1);
                dadosCarregados.Add(item);
            }
            catch { /* IGNORA */ }
        }
    }
    
    // formata a string para float (caso haja erro)
    private float ParseFloat(string val)
    {
        if (float.TryParse(val, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out float result))
            return result;
        return 0.0f;
    }

    private void PlotarGrafico(List<double> real, List<double> predito)
    {
        // painel do gráfico
        GraphPane pane = zedGraphControl1.GraphPane;
        pane.CurveList.Clear();
        
        // cria as listas de pontos 
        PointPairList listReal = new PointPairList();
        PointPairList listPred = new PointPairList();

        // adiciona os pontos
        for (int i = 0; i < real.Count; i++)
        {
            listReal.Add(i, real[i]);
            if (i < predito.Count) listPred.Add(i, predito[i]);
        }
        
        // adiciona a curva real
        LineItem curvaReal = pane.AddCurve("Real", listReal, Color.Blue, SymbolType.None);
        curvaReal.Line.Width = 2;

        // adiciona a curva predita
        LineItem curvaPred = pane.AddCurve("IA Predição", listPred, Color.Red, SymbolType.None);
        curvaPred.Line.Width = 2;
        curvaPred.Line.Style = System.Drawing.Drawing2D.DashStyle.Dash; // linha tracejada
        
        zedGraphControl1.AxisChange();
        zedGraphControl1.Invalidate();
    }
    
}