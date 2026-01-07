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
        
        public static class Scaler
        {
            // Valores copiados do arquivo
            
            public static double[] Means = new double[] {
                539.141513, // PEHIST (Índice 0)
                4.150937,   // PSHIST (Índice 1)
                0.589809,   // REGULADOR1 (Índice 2)
                0.478988,   // REGULADOR2 (Índice 3)
                584.288289, // PDT1 (Índice 4)
                576.223849, // PDT2 (Índice 5)
                12.383923,  // FT1 (Índice 6)
                -0.000100,  // PEHIST_delta1
                -0.000103,  // PEHIST_delta2
                -0.000003,  // PSHIST_delta1
                -0.000002   // PSHIST_delta2
            };

            // Lista 'scale' do arquivo
            public static double[] Scales = new double[] {
                110.198308, // PEHIST
                0.334460,   // PSHIST
                0.427771,   // REGULADOR1
                0.435729,   // REGULADOR2
                78.077699,  // PDT1
                85.068249,  // PDT2
                5.801659,   // FT1
                0.245084,   // PEHIST_delta1
                0.252037,   // PEHIST_delta2
                0.003503,   // PSHIST_delta1
                0.003569    // PSHIST_delta2
            };

            // Fórmula do StandardScaler Inverso: Real = (ValorIA * Scale) + Mean
            public static double Desnormalizar(double valorIA, int indiceColuna)
            {
                return (valorIA * Scales[indiceColuna]) + Means[indiceColuna];
            }
            
            public static float Normalizar(float valorReal, int indiceColuna)
            {
                return (float)((valorReal - Means[indiceColuna]) / Scales[indiceColuna]);
            }
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
            // O ARQUIVO EXIGE 512. NÃO PODEMOS MUDAR ISSO NO CÓDIGO.
            const int TAMANHO_JANELA_MODELO = 512;
            
            // O amigo analisa 336. Vamos usar isso apenas para o gráfico.
            const int TAMANHO_JANELA_GRAFICO = 336; 

            if (dadosCsvRaw.Count == 0 || string.IsNullOrEmpty(path_ONNX))
            {
                MessageBox.Show("Carregue CSV e ONNX primeiro.");
                return;
            }

            try
            {
                this.Text = "Processando... (Aguarde)";
                Application.DoEvents();

                // 1. SELEÇÃO DE DADOS
                // Precisamos de pelo menos 512 dados para o modelo rodar.
                // Pegamos 1000 para ter folga.
                List<DadosCSV> dadosParaProcessar;
                
                if (dadosCsvRaw.Count > 1000)
                    dadosParaProcessar = dadosCsvRaw.Skip(dadosCsvRaw.Count - 1000).ToList();
                else
                    dadosParaProcessar = dadosCsvRaw.ToList();

                // Prepara os dados com janela de 512 (Obrigatório pelo erro que deu)
                var dadosInput = PrepararDados(dadosParaProcessar, TAMANHO_JANELA_MODELO);

                MLContext mlContext = new MLContext();
                IDataView dataView = mlContext.Data.LoadFromEnumerable(dadosInput);

                // 2. SHAPE DICTIONARY (VOLTA PARA 512)
                var shapeDict = new Dictionary<string, int[]>()
                {
                    { "batch_x", new[] { 1, TAMANHO_JANELA_MODELO, 7 } },
                    { "batch_x_mark", new[] { 1, TAMANHO_JANELA_MODELO, 5 } },
                    { "batch_x_aux", new[] { 1, TAMANHO_JANELA_MODELO, 4 } }
                };

                var pipeline = mlContext.Transforms.ApplyOnnxModel(
                    modelFile: path_ONNX,
                    outputColumnNames: new[] { "output" },
                    inputColumnNames: new[] { "batch_x", "batch_x_mark", "batch_x_aux" },
                    shapeDictionary: shapeDict,
                    gpuDeviceId: null,
                    fallbackToCpu: true
                );

                var transformer = pipeline.Fit(dataView);
                var transformedData = transformer.Transform(dataView);
                var predictions = mlContext.Data.CreateEnumerable<OnnxOutput>(transformedData, reuseRowObject: false).ToList();

                if (predictions.Count == 0 || predictions[0].PredictedValue == null)
                {
                    MessageBox.Show("Erro: A IA não retornou dados.");
                    return;
                }

                // --- 3. FILTRAGEM PARA O GRÁFICO ---
                // A IA devolveu vetores de 512 posições.
                // Mas nós só queremos ver as últimas 336.
                
                // Pegamos o final das listas de dados originais
                var dadosFinaisReal = dadosParaProcessar.Skip(dadosParaProcessar.Count - TAMANHO_JANELA_GRAFICO).ToList();
                // Pegamos o final das predições
                var predicoesFinais = predictions.Skip(predictions.Count - TAMANHO_JANELA_GRAFICO).ToList();

                int indiceEscolhido = comboBox1.SelectedIndex;
                if (indiceEscolhido < 0) indiceEscolhido = 0;

                List<double> yReal = new List<double>();
                List<double> yPredito = new List<double>();

                for(int i = 0; i < dadosFinaisReal.Count; i++)
                {
                    // Real
                    yReal.Add(IndiceSwitch(dadosFinaisReal[i], indiceEscolhido));

                    // Predito
                    if (i < predicoesFinais.Count && predicoesFinais[i].PredictedValue != null)
                    {
                        // O array PredictedValue tem 512 itens (0 a 511).
                        // O item [0] é a previsão da primeira coluna (PEHIST).
                        // O item [1] é a previsão da segunda coluna (PSHIST).
                        // ISSO NÃO MUDA COM O TEMPO. A IA devolve APENAS o passo atual.
                        
                        double valorBruto = predicoesFinais[i].PredictedValue[indiceEscolhido];
                        double valorReal = Scaler.Desnormalizar(valorBruto, indiceEscolhido);
                        yPredito.Add(valorReal);
                    }
                    else
                    {
                        yPredito.Add(0);
                    }
                }

                PlotarGrafico(yReal, yPredito);
                this.Text = $"Concluído! (Janela Modelo: 512 | Visualizado: 336)";
                MessageBox.Show($"Sucesso! O modelo rodou com 512, mas mostramos os últimos 336 pontos.");
            }
            catch (Exception ex)
            {
                MessageBox.Show($"ERRO:\n{ex.Message}");
            }
        }

        private List<OnnxInputPronto> PrepararDados(List<DadosCSV> raw, int janela)
        {
            var listaPronta = new List<OnnxInputPronto>();

            for (int i = 0; i < raw.Count; i++)
            {
                var input = new OnnxInputPronto();

                List<float> vetorX = new List<float>(); 
                List<float> vetorAux = new List<float>(); 
                List<float> vetorMark = new List<float>(); 

                for (int j = 0; j < janela; j++)
                {
                    int indiceNoPassado = i - (janela - 1) + j;
                    if (indiceNoPassado < 0) indiceNoPassado = 0;

                    var linha = raw[indiceNoPassado];

                    // --- INPUT 1: BATCH_X (Normalizado pelos índices 0 a 6) ---
                    vetorX.Add(Scaler.Normalizar(linha.PEHIST, 0));
                    vetorX.Add(Scaler.Normalizar(linha.PSHIST, 1));
                    vetorX.Add(Scaler.Normalizar(linha.REGULADOR1, 2));
                    vetorX.Add(Scaler.Normalizar(linha.REGULADOR2, 3));
                    vetorX.Add(Scaler.Normalizar(linha.PDT1, 4));
                    vetorX.Add(Scaler.Normalizar(linha.PDT2, 5));
                    vetorX.Add(Scaler.Normalizar(linha.FT1, 6));

                    // --- INPUT 2: BATCH_X_AUX (Normalizado pelos índices 7 a 10) ---
                    vetorAux.Add(Scaler.Normalizar(linha.PEHIST_smooth_delta1, 7));
                    vetorAux.Add(Scaler.Normalizar(linha.PEHIST_smooth_delta2, 8));
                    vetorAux.Add(Scaler.Normalizar(linha.PSHIST_smooth_delta1, 9));
                    vetorAux.Add(Scaler.Normalizar(linha.PSHIST_smooth_delta2, 10));

                    DateTime dataAtual = DateTime.Parse(linha.Date);

                    float month = (dataAtual.Month - 1) / 11.0f - 0.5f;
                    float day = (dataAtual.Day - 1) / 30.0f - 0.5f;     // Aproximação
                    float weekday = (int)dataAtual.DayOfWeek / 6.0f - 0.5f;
                    float hour = dataAtual.Hour / 23.0f - 0.5f;
                    
                    float minuteRaw = (float)(dataAtual.Minute / 15); 
                    float minute = minuteRaw / 3.0f - 0.5f; 

                    vetorMark.Add(month);
                    vetorMark.Add(day);
                    vetorMark.Add(weekday);
                    vetorMark.Add(hour);
                    vetorMark.Add(minute);
                }

                input.BatchX = vetorX.ToArray(); 
                input.BatchXAux = vetorAux.ToArray(); 
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
            // ... Dentro de PlotarGrafico ...

            // curva predito (IA) 
            LineItem curvaPred = pane.AddCurve("IA Predição", listPred, Color.Orange, SymbolType.None);
            curvaPred.Line.Width = 2;
            curvaPred.Line.Style = System.Drawing.Drawing2D.DashStyle.Solid;
            curvaPred.IsY2Axis = false; 

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