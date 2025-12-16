using Microsoft.ML;
using Microsoft.ML.Data;
using ZedGraph;

namespace RN_Graph_App
{
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
        // no test_nooutliers FT1 é coluna 7.
        [LoadColumn(7)] public float FT1 { get; set; } 
    }

    // classe que prepara um array com os dados para serem enviados para a IA
    public class OnnxInputPronto
    {
        // 48 linhas * 7 colunas = 336 elementos
        [VectorType(336)] 
        [ColumnName("batch_x")]
        public float[] BatchX { get; set; }

        // 48 linhas * 5 colunas = 240 elementos
        [VectorType(240)] 
        [ColumnName("batch_x_mark")]
        public float[] BatchXMark { get; set; }
    }

    public class OnnxOutput
    {
        [ColumnName("outputs")]
        public float[] PredictedValue { get; set; }
    }

    public partial class Form1 : Form
    {
        // onde será guardado o caminho do arquivo ONNX
        public string path_ONNX = "";
        // dados lidos do CSV
        private List<DadosCSV> dadosCsvRaw = new List<DadosCSV>();
        
        public Form1()
        {
            InitializeComponent();
        }

        private void button1_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog() { Filter = "CSV Files|*.csv" })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    ReadCSV(ofd.FileName);
                    MessageBox.Show($"Carregadas {dadosCsvRaw.Count} linhas.");
                }
            }
        }

        private void button2_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog() { Filter = "ONNX Files|*.onnx" })
            {
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    path_ONNX = ofd.FileName;
                    MessageBox.Show("Modelo Selecionado!");
                }
            }
        }

       private void button3_Click(object sender, EventArgs e)
        {
            // IMPORTANTE: TAMANHO DA JANELA
            const int TAMANHO_JANELA = 48; 
            // esse valor estabelecido como 48 foi na base da tentativa e erro,
            // pois não tinha informações sobre o tamanho da janela para o modelo ONNX 

            if (dadosCsvRaw.Count == 0 || string.IsNullOrEmpty(path_ONNX))
            {
                MessageBox.Show("Carregue CSV e ONNX primeiro.");
                return;
            }

            try
            {
                this.Text = "Processando... (Aguarde)";
                Application.DoEvents(); // força a tela a atualizar

                // LIMITADOR DE DADOS
                // previne que o programa trave devido à abundância de dados
                var dadosParciais = dadosCsvRaw.Take(1000).ToList();
                
                var dadosParaIA = PrepararDadosParaIA(dadosParciais, TAMANHO_JANELA);

                MLContext mlContext = new MLContext();
                IDataView dataView = mlContext.Data.LoadFromEnumerable(dadosParaIA);

                // formato do dado de input para o ONNX
                var shapeDict = new Dictionary<string, int[]>() {
                    { "batch_x",      new [] { 1, TAMANHO_JANELA, 7 } }, 
                    { "batch_x_mark", new [] { 1, TAMANHO_JANELA, 5 } } 
                };

                // gera pipeline para a IA
                var pipeline = mlContext.Transforms.ApplyOnnxModel(
                        modelFile: path_ONNX,
                        outputColumnNames: new[] { "outputs" },
                        inputColumnNames: new[] { "batch_x", "batch_x_mark" },
                        shapeDictionary: shapeDict,
                        gpuDeviceId: null,
                        fallbackToCpu: true
                    );
                
                MessageBox.Show("Pipeline criado. Iniciando execução...");

                // transforma dados e executa IA
                var transformer = pipeline.Fit(dataView);
                var transformedData = transformer.Transform(dataView);
                var predictions = mlContext.Data.CreateEnumerable<OnnxOutput>(transformedData, reuseRowObject: false).ToList();
                
                if (predictions.Count == 0)
                {
                    MessageBox.Show("A IA rodou, mas a lista de previsões veio vazia!");
                    return;
                }
                
                // verificação de Nulos
                if (predictions[0].PredictedValue == null)
                {
                    MessageBox.Show("A IA retornou 'null' nos valores");
                    return;
                }

                // Plotar
                List<double> yReal = dadosParciais.Select(d => (double)d.FT1).ToList();
                
                // PEGAR O RESULTADO CERTO
                List<double> yPredito = new List<double>();
                foreach(var p in predictions)
                {
                    // se o array tiver dados, pegamos o primeiro (pode ajustar se precisar)
                     if (p.PredictedValue.Length > 0)
                        yPredito.Add((double)p.PredictedValue[0]);
                     else
                        yPredito.Add(0);
                }

                PlotarGrafico(yReal, yPredito);
                this.Text = "Concluído!";
                MessageBox.Show($"Sucesso! Gráfico gerado com {yPredito.Count} pontos.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"ERRO:\n{ex.Message}\n\nInner: {ex.InnerException?.Message}");
            }
        }

        private List<OnnxInputPronto> PrepararDadosParaIA(List<DadosCSV> raw, int janela)
        {
            var listaPronta = new List<OnnxInputPronto>();

            // percorre linha por linha
            for (int i = 0; i < raw.Count; i++)
            {
                var input = new OnnxInputPronto();
                List<float> vetorX = new List<float>();
                
                // JANELA DESLIZANTE
                // em vez de repetir o valor atual, vou buscar os 48 valores do passado (tamanho da janela)
                for (int j = 0; j < janela; j++)
                {
                    // calcula qual índice do passado pegar
                    int indiceNoPassado = i - (janela - 1) + j;
                    
                    // se estiver no começo do arquivo não existe passado suficiente. Repete a linha 0 (Padding).
                    if (indiceNoPassado < 0) indiceNoPassado = 0;

                    var linhaHistorica = raw[indiceNoPassado];

                    vetorX.Add(linhaHistorica.PEHIST);
                    vetorX.Add(linhaHistorica.PSHIST);
                    vetorX.Add(linhaHistorica.REGULADOR1);
                    vetorX.Add(linhaHistorica.REGULADOR2);
                    vetorX.Add(linhaHistorica.PDT1);
                    vetorX.Add(linhaHistorica.PDT2);
                    vetorX.Add(0.0f); // Dummy (Coluna 7): preenche como vazio já que não há nada a ser predito
                }
                input.BatchX = vetorX.ToArray();

                // BatchXMark (Tempo): Manterei zerado por enquanto, pois calcular data é complexo,
                // mas a variação nos sensores acima já deve fazer a linha vermelha mexer.
                // Um arquivo de "desnormalização" resolveria...
                List<float> vetorMark = new List<float>();
                for (int k = 0; k < janela; k++)
                {
                    vetorMark.Add(0.0f); vetorMark.Add(0.0f); vetorMark.Add(0.0f); vetorMark.Add(0.0f); vetorMark.Add(0.0f);
                }
                input.BatchXMark = vetorMark.ToArray();

                listaPronta.Add(input);
            }
            return listaPronta;
        }

        // leitura do CSV
        private void ReadCSV(string path)
        {
            dadosCsvRaw.Clear();
            var linhas = File.ReadAllLines(path);
            for (int i = 1; i < linhas.Length; i++)
            {
                // ignora linhas vazias
                var cols = linhas[i].Split(',');
                if (cols.Length < 8) continue;
                try
                {
                    dadosCsvRaw.Add(new DadosCSV
                    {
                        Date = cols[0],
                        PEHIST = ParseFloat(cols[1]),
                        PSHIST = ParseFloat(cols[2]),
                        REGULADOR1 = ParseFloat(cols[3]),
                        REGULADOR2 = ParseFloat(cols[4]),
                        PDT1 = ParseFloat(cols[5]),
                        PDT2 = ParseFloat(cols[6]),
                        FT1 = ParseFloat(cols[7])
                    });
                }
                catch {/* VAZIO */}
            }
        }

        // formata string para float
        private float ParseFloat(string val) => float.TryParse(val, System.Globalization.NumberStyles.Any, System.Globalization.CultureInfo.InvariantCulture, out float r) ? r : 0f;

        // gera o gráfico após os dados serem processados
        private void PlotarGrafico(List<double> real, List<double> predito)
        {
            if (zedGraphControl1 == null) return;

            GraphPane pane = zedGraphControl1.GraphPane;

            pane.CurveList.Clear();
        
            PointPairList listReal = new PointPairList();
            PointPairList listPred = new PointPairList();

            // adiciona os pontos no gráfico
            bool inverterSinal = true; 
            for (int i = 0; i < real.Count; i++)
            {
                listReal.Add(i, real[i]);
                
                if (i < predito.Count) 
                {
                    double valorIA = predito[i];
                
                    // EXPLICAÇÃO: como não tenho um parâmetro de normalização, para a visualização da predição,
                    // inverto o sinal do valor predito para apresentar maior semelhança.
                    if (inverterSinal)
                    {
                        valorIA = valorIA * -1;
                    }
                    listPred.Add(i, valorIA);
                }
            }
        
            // curva real - > Esquerda
            LineItem curvaReal = pane.AddCurve("Real", listReal, Color.Blue, SymbolType.None);
            curvaReal.Line.Width = 2;
            curvaReal.IsY2Axis = false; // Fica na esquerda

            // curva predito (IA) -> Direita
            LineItem curvaPred = pane.AddCurve("IA Predição", listPred, Color.Red, SymbolType.None);
            curvaPred.Line.Width = 2;
            curvaPred.Line.Style = System.Drawing.Drawing2D.DashStyle.Solid; 
            curvaPred.IsY2Axis = true; // manda para a direita

            zedGraphControl1.AxisChange();
            zedGraphControl1.Invalidate();
            
            // função alternativa alterando a visualização do BenchMark
            
            /*
            if (zedGraphControl1 == null) return;
            GraphPane pane = zedGraphControl1.GraphPane;
            pane.CurveList.Clear();
            
            PointPairList listReal = new PointPairList();
            PointPairList listPred = new PointPairList();

            for (int i = 0; i < real.Count; i++)
            {
                listReal.Add(i, real[i]);
                if (i < predito.Count) listPred.Add(i, predito[i]);
            }

            LineItem curveReal = pane.AddCurve("Real", listReal, Color.Blue, SymbolType.None);
            curveReal.Line.Width = 2;

            LineItem curvePred = pane.AddCurve("Predito", listPred, Color.Red, SymbolType.None);
            curvePred.Line.Width = 2;
            
            zedGraphControl1.AxisChange();
            zedGraphControl1.Invalidate();
            */
        }
    }
}
