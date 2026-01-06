using Microsoft.ML;
using Microsoft.ML.Data;
using ZedGraph;

namespace RN_Graph_App
{

    // classe que prepara um array com os dados para serem enviados para a IA
    public class OnnxInputPronto
    {
        // tensor: float32[batch_size,512,7] -> 512 x 7 = 3584
        [VectorType(3584)] 
        [ColumnName("batch_x")]
        public float[] BatchX { get; set; }

        // tensor: float32[batch_size,512,5] -> 512 x 5 = 2560
        [VectorType(2560)] 
        [ColumnName("batch_x_mark")]
        public float[] BatchXMark { get; set; }
        
        // tensor: float32[batch_size,512,4] -> 512 x 4 = 2048
        [VectorType(2048)] 
        [ColumnName("batch_x_aux")]
        public float[] BatchXAux { get; set; }
    }

    public class OnnxOutput
    {
        [ColumnName("output")]
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
            // tamanho esperado pelo modelo
            const int TAMANHO_JANELA = 512;

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
                var dadosParciais = dadosCsvRaw.Take(336).Reverse().ToList();
                dadosParciais.Reverse();

                var dadosParaIA = PrepararDados(dadosParciais, TAMANHO_JANELA);

                MLContext mlContext = new MLContext();
                IDataView dataView = mlContext.Data.LoadFromEnumerable(dadosParaIA);

                // formato do dado de input para o ONNX
                var shapeDict = new Dictionary<string, int[]>()
                {
                    { "batch_x", new[] { 1, TAMANHO_JANELA, 7 } },
                    { "batch_x_mark", new[] { 1, TAMANHO_JANELA, 5 } },
                    { "batch_x_aux", new[] { 1, TAMANHO_JANELA, 4 } }
                };

                // gera pipeline para a IA
                var pipeline = mlContext.Transforms.ApplyOnnxModel(
                    modelFile: path_ONNX,
                    outputColumnNames: new[] { "output" },
                    inputColumnNames: new[] { "batch_x", "batch_x_mark", "batch_x_aux" },
                    shapeDictionary: shapeDict,
                    gpuDeviceId: null,
                    fallbackToCpu: true
                );

                MessageBox.Show("Pipeline criado. Iniciando execução...");

                // transforma dados e executa IA
                var transformer = pipeline.Fit(dataView);
                var transformedData = transformer.Transform(dataView);
                var predictions = mlContext.Data.CreateEnumerable<OnnxOutput>(transformedData, reuseRowObject: false)
                    .ToList();

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

                int indice = comboBox1.SelectedIndex;
                if (indice < 0) indice = 0; // proteção
                
                // listas para o gráfico
                List<double> yReal = new List<double>();
                List<double> yPredito = new List<double>();
                
                int totalItens = dadosParciais.Count;

                for (int i = 0; i < totalItens; i++)
                {
                    // pega o valor real do dado
                    double valorReal = IndiceSwitch(dadosParciais[i], indice);
                    yReal.Add(valorReal);
                    
                    // pega o valor predito da IA
                    var p = predictions[i];
                    if (p.PredictedValue != null && p.PredictedValue.Length > indice)
                    {
                        yPredito.Add((double)p.PredictedValue[indice]);
                    }
                    else
                    {
                        yPredito.Add(0);
                    }
                }
                PlotarGrafico(yReal, yPredito);
                this.Text = $"Concluído! Visualizando: {comboBox1.SelectedItem}";
                MessageBox.Show($"Gráfico gerado para a coluna: {comboBox1.SelectedItem}");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"ERRO:\n{ex.Message}\n\nInner: {ex.InnerException?.Message}");
            }
        }

        private List<OnnxInputPronto> PrepararDados(List<DadosCSV> raw, int janela)
        {
            var listaPronta = new List<OnnxInputPronto>();

            // percorre linha por linha
            for (int i = 0; i < raw.Count; i++)
            {
                var input = new OnnxInputPronto();

                // listas separadas para cada entrada do ONNX
                List<float> vetorX = new List<float>(); // Para batch_x (7 features)
                List<float> vetorAux = new List<float>(); // Para batch_x_aux (4 features)
                List<float> vetorMark = new List<float>(); // Para batch_x_mark (5 features)

                // JANELA DESLIZANTE
                // em vez de repetir o valor atual, vou buscar os 512 valores do passado (tamanho da janela)
                for (int j = 0; j < janela; j++)
                {
                    // calcula qual índice do passado pegar
                    int indiceNoPassado = i - (janela - 1) + j;

                    // se estiver no começo do arquivo não existe passado suficiente. Repete a linha 0 (Padding).
                    if (indiceNoPassado < 0) indiceNoPassado = 0;

                    var linha = raw[indiceNoPassado];

                    // INPUT 1: batch_x (7 colunas principais)
                    vetorX.Add(linha.PEHIST);
                    vetorX.Add(linha.PSHIST);
                    vetorX.Add(linha.REGULADOR1);
                    vetorX.Add(linha.REGULADOR2);
                    vetorX.Add(linha.PDT1);
                    vetorX.Add(linha.PDT2);
                    vetorX.Add(linha.FT1);

                    // INPUT 2: batch_x_aux (4 colunas delta)
                    vetorAux.Add(linha.PEHIST_smooth_delta1);
                    vetorAux.Add(linha.PEHIST_smooth_delta2);
                    vetorAux.Add(linha.PSHIST_smooth_delta1);
                    vetorAux.Add(linha.PSHIST_smooth_delta2);

                    // INPUT 3: batch_x_mark (5 dummys de tempo)
                    // 5 zeros por linha
                    vetorMark.Add(0.0f);
                    vetorMark.Add(0.0f);
                    vetorMark.Add(0.0f);
                    vetorMark.Add(0.0f);
                    vetorMark.Add(0.0f);
                }

                // converte as listas para Arrays e atribui ao objeto
                input.BatchX = vetorX.ToArray(); // Tamanho 3584 (512*7)
                input.BatchXAux = vetorAux.ToArray(); // Tamanho 2048 (512*4)
                input.BatchXMark = vetorMark.ToArray(); // Tamanho 2560 (512*5)

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
                // de 0 a 11
                if (cols.Length < 12) continue;
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
                        FT1 = ParseFloat(cols[7]),
                        PEHIST_smooth_delta1 = ParseFloat(cols[8]),
                        PEHIST_smooth_delta2 = ParseFloat(cols[9]),
                        PSHIST_smooth_delta1 = ParseFloat(cols[10]),
                        PSHIST_smooth_delta2 = ParseFloat(cols[11])
                    });
                }
                catch
                {
                    /* VAZIO */
                }
            }
        }

        // formata string para float
        private float ParseFloat(string val) => float.TryParse(val, System.Globalization.NumberStyles.Any,
            System.Globalization.CultureInfo.InvariantCulture, out float r)
            ? r
            : 0f;

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

        // função auxiliar de Switch para obter o índice 
        private double IndiceSwitch(DadosCSV dado, int indice)
        {
            switch (indice)
            {
                case 0: return (double)dado.PEHIST;
                case 1: return (double)dado.PSHIST;
                case 2: return (double)dado.REGULADOR1;
                case 3: return (double)dado.REGULADOR2;
                case 4: return (double)dado.PDT1;
                case 5: return (double)dado.PDT2;
                case 6: return (double)dado.FT1;
                default: return 0;
            }
        }
    }
}