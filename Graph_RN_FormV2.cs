using System.Globalization;
using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using RN_Graph_App.Models;
using ZedGraph;

namespace WinFormsOnnxApp
{
    public partial class Graph_RN_FormV2 : Form
    {
        // CONFIGURAÇÕES DO MODELO
        private const int SeqLen = 512; // janela de imput
        private const int PredLen = 336; // janela da predição
        
        // definição das features (batch_x)
        private const int FeatX = 7;
        private readonly string[] _featureNames = new string[] 
        { 
            "PEHIST", "PSHIST", "REGULADOR1", "REGULADOR2", "PDT1", "PDT2", "FT1" 
        };

        private const int FeatAux = 4;
        private const int FeatMark = 5;

        // path para os arquivos
        private string _csvPath;
        private string _onnxPath = "final_model.onnx"; // ultimo modelo mandado como base
        private List<TimeSeriesPoint> _allData;
        
        // Scalers (Média e Desvio Padrão)
        private double[] _meanX, _stdX;
        private double[] _meanAux, _stdAux;

        // armazena resultado da IA e índice de onde começou o teste
        private Tensor<float> _lastOutputTensor;
        private int _lastStartIndex = -1;
        
        // VARIÁVEL PARA TODOS OS LOTES: lista para armazenar os blocos de predição
        private List<float[]> _predictionHistory;

        public Graph_RN_FormV2()
        {
            InitializeComponent();
            SetupGraph();
            
            // ComboBox das colunas
            cmbColumns.Items.AddRange(_featureNames);
            cmbColumns.SelectedIndex = 1; // PSHIST como padrão
            cmbColumns.Enabled = false;
            
            // ComboBox de modos de visualização
            cmbViewMode.Items.Add("Validação (336 Steps)");
            cmbViewMode.Items.Add("Histórico completo (Tempo)");
            cmbViewMode.SelectedIndex = 0; // padrão: validação
            cmbViewMode.SelectedIndexChanged += CmbViewMode_SelectedIndexChanged;
            cmbViewMode.Enabled = false;
        }

        // configuração padrão do gráfico
        private void SetupGraph()
        {
            GraphPane myPane = zedGraphControl1.GraphPane;
            myPane.Title.Text = "Validação: Real vs Predição";
            myPane.XAxis.Title.Text = "Horizonte de previsão (Steps)";
            myPane.YAxis.Title.Text = "Valor";
            
            // grades para facilitar leitura
            myPane.XAxis.MajorGrid.IsVisible = true;
            myPane.YAxis.MajorGrid.IsVisible = true;
            myPane.XAxis.MajorGrid.Color = Color.LightGray;
            myPane.YAxis.MajorGrid.Color = Color.LightGray;
        }
        
        // evento que troca o modo de visualização
        private void CmbViewMode_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (_lastOutputTensor != null) UpdatePlot();
        }
        
        // botão de leitura do CSV
        private void btnLoadData_Click(object sender, EventArgs e)
        {
            using (OpenFileDialog ofd = new OpenFileDialog())
            {
                ofd.Filter = "CSV Files|*.csv";
                if (ofd.ShowDialog() == DialogResult.OK)
                {
                    _csvPath = ofd.FileName;
                    LoadCsv(_csvPath);
                }
            }
        }

        // carrega dados do csv
        private void LoadCsv(string path)
        {
            try
            {
                // le todas as linhas do arquivo
                var lines = File.ReadAllLines(path);
                _allData = new List<TimeSeriesPoint>();

                for (int i = 1; i < lines.Length; i++)
                {
                    // quebra de linhas (parts) na vírgula
                    var parts = lines[i].Split(',');
                    if (parts.Length < 12) continue; // verifica se tem 12 colunas
                    
                    TimeSeriesPoint row = new TimeSeriesPoint();
                    // tenta ler a primeira coluna como data
                    if (DateTime.TryParse(parts[0], CultureInfo.InvariantCulture, DateTimeStyles.None, out DateTime dt))
                        row.Date = dt;
                    else 
                        row.Date = DateTime.MinValue;
                    
                    // preenche o vetor X (batch_x)
                    row.X = new float[FeatX];
                    for (int j = 0; j < FeatX; j++) row.X[j] = ParseFloat(parts[j + 1]);

                    // preenche o vetor Aux (batch_aux)
                    row.Aux = new float[FeatAux];
                    for (int j = 0; j < FeatAux; j++) row.Aux[j] = ParseFloat(parts[j + 1 + FeatX]);

                    _allData.Add(row);
                }
                
                // verifica se o arquivo csv tem tamanho suficiente
                if (_allData.Count < SeqLen + PredLen)
                {
                    MessageBox.Show($"CSV insuficiente. Mínimo: {SeqLen + PredLen} linhas.");
                    return;
                }
                
                // CALCULA A MEDIA E DESVIO PADRÃO
                FitScalers();
                lblStatus.Text = $"Dados carregados: {_allData.Count} linhas.";
                btnRunInference.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Erro ao ler CSV: " + ex.Message);
            }
        }
        
        // formatação de valores para float 
        private float ParseFloat(string val)
        {
            if (float.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out float res)) return res;
            return 0f;
        }

        // função de cálculo de média e variação
        private void FitScalers()
        {
            _meanX = new double[FeatX]; // vetor de média das colunas
            _stdX = new double[FeatX]; // vetor de desvio padrão
            _meanAux = new double[FeatAux];
            _stdAux = new double[FeatAux];

            int n = _allData.Count;

            // calculo de média aritmética
            foreach (var row in _allData)
            {
                for (int i = 0; i < FeatX; i++) _meanX[i] += row.X[i]; // soma todos os valores 
                for (int i = 0; i < FeatAux; i++) _meanAux[i] += row.Aux[i]; // soma os aux
            }
            // divide pelo número de linhas 
            for (int i = 0; i < FeatX; i++) _meanX[i] /= n;
            for (int i = 0; i < FeatAux; i++) _meanAux[i] /= n;

            // calculo do desvio padrão
            foreach (var row in _allData)
            {
                // dist de cada ponto até a média, elevada ao quadrado
                for (int i = 0; i < FeatX; i++) _stdX[i] += Math.Pow(row.X[i] - _meanX[i], 2);
                for (int i = 0; i < FeatAux; i++) _stdAux[i] += Math.Pow(row.Aux[i] - _meanAux[i], 2);
            }
            // tira raiz quadrada da média
            for (int i = 0; i < FeatX; i++) 
            {
                _stdX[i] = Math.Sqrt(_stdX[i] / n);
                if (_stdX[i] == 0) _stdX[i] = 1; // não deixa ocorrer divisão por zero
            }
            for (int i = 0; i < FeatAux; i++)
            {
                _stdAux[i] = Math.Sqrt(_stdAux[i] / n);
                if (_stdAux[i] == 0) _stdAux[i] = 1;
            }
        }

        private void btnRunInference_Click(object sender, EventArgs e)
        {
            // verifica se o arquivo onnx existe
            // caso não existir, abre a janela de procura
            if (!File.Exists(_onnxPath))
            {
                using (OpenFileDialog ofd = new OpenFileDialog())
                {
                    ofd.Filter = "ONNX Model|*.onnx";
                    if (ofd.ShowDialog() == DialogResult.OK) _onnxPath = ofd.FileName;
                    else return;
                }
            }

            try
            {
                /* =========== PARTE DE PREDIÇÃO E INDEX ===============
                 
                pred_len = 336 -> tamanho da predição
                preds_pytorch[0] -> primeira amostra do conjunto
                int(len(df_raw) * 0.1) -> 10% dos dados reais
                border1s[2] -> índice da janela inicial
            
                border1s[2] = Total - num_test - seq_len
                
                
                 ===================================================== */
                
                // Backtesting: separa os últimos x dados reais (predição)
                int totalCount = _allData.Count;
                int numTest = (int)(totalCount * 0.1); // 10% para teste
                
                // [cite_start]
                // fórmula: len(df) - num_test - seq_len
                _lastStartIndex = totalCount - numTest - SeqLen;
                // indice onde começa a leitura dos 512 dados de entrada
                
                if (_lastStartIndex < 0)
                {
                    MessageBox.Show("Dados insuficientes.");
                    return;
                }
                
                // inicia a lista de histórico
                _predictionHistory = new List<float[]>();

                using (var session = new InferenceSession(_onnxPath))
                {
                    for (int i = _lastStartIndex; i <= totalCount - SeqLen; i += PredLen)
                    {
                        // prepara dados para essa janela que percorrerá o input
                        var inputWindow = _allData.GetRange(i, SeqLen);
                        
                        // passagem de dados pela rede neural
                        var inputs = PrepareInputs(inputWindow);

                        using (var results = session.Run(inputs))
                        {
                            // armazena os dados de output em memória 
                            var outputRaw = results.First(x => x.Name == "output").AsTensor<float>();
                            float[] batchResult = outputRaw.ToArray();
                            
                            // salva o resultado desse lote na lista
                            _predictionHistory.Add(batchResult);
                        }
                    }
                }
                
                // armazena o primeiro tensor
                if (_predictionHistory.Count > 0)
                {
                    long[] dims = { 1, PredLen, 7 }; // dimensões fixas do modelo
                    _lastOutputTensor = new DenseTensor<float>(_predictionHistory[0], new ReadOnlySpan<int>(new[] { 1, PredLen, 7 }));
                }

                cmbColumns.Enabled = true;
                cmbViewMode.Enabled = true;
                // desenha as linhas no gráfico
                UpdatePlot(); 
                lblStatus.Text = "Gráfico gerado. Selecione o modo de visualização.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Erro na inferência: {ex.Message}");
            }
        }
        
        // função auxiliar para preparar os tensores
        private List<NamedOnnxValue> PrepareInputs(List<TimeSeriesPoint> window)
        {
            // prepara os tensores
            var tensorX = new DenseTensor<float>(new[] { 1, SeqLen, FeatX });
            var tensorMark = new DenseTensor<float>(new[] { 1, SeqLen, FeatMark });
            var tensorAux = new DenseTensor<float>(new[] { 1, SeqLen, FeatAux });

            for (int t = 0; t < SeqLen; t++)
            {
                var row = window[t];

                // normalizar batch_x
                for (int f = 0; f < FeatX; f++)
                    // subtrai a média e dividimos pelo desvio padrão
                    tensorX[0, t, f] = (float)((row.X[f] - _meanX[f]) / _stdX[f]);

                // normalizar o temporal
                tensorMark[0, t, 0] = (float)((row.Date.Month - 1) / 11.0 - 0.5);
                tensorMark[0, t, 1] = (float)((row.Date.Day - 1) / 30.0 - 0.5);
                tensorMark[0, t, 2] = (float)((int)row.Date.DayOfWeek / 6.0 - 0.5);
                tensorMark[0, t, 3] = (float)(row.Date.Hour / 23.0 - 0.5);
                tensorMark[0, t, 4] = (float)(row.Date.Minute / 59.0 - 0.5);

                // normalizar aux
                for (int f = 0; f < FeatAux; f++)
                    tensorAux[0, t, f] = (float)((row.Aux[f] - _meanAux[f]) / _stdAux[f]);
            }

            // cira pacote com os tensores criados, com os nomes esperados
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("batch_x", tensorX),
                NamedOnnxValue.CreateFromTensor("batch_x_mark", tensorMark),
                NamedOnnxValue.CreateFromTensor("batch_x_aux", tensorAux)
            };
        }

        // comboBox para trocar de gráfico de acordo com a coluna 
        private void cmbColumns_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (_lastOutputTensor != null)
            {
                UpdatePlot();
            }
        }

        private void UpdatePlot()
        {
            // verifica se a IA já rodou
            if (_predictionHistory == null || _predictionHistory.Count == 0) return;
            
            // verifica a coluna escolhida
            int selectedFeatureIndex = cmbColumns.SelectedIndex;
            if (selectedFeatureIndex < 0) selectedFeatureIndex = 0;
            
            // verifica o modo de vizualização
            int viewMode = cmbViewMode.SelectedIndex;
            
            // limpa o gráfico
            GraphPane pane = zedGraphControl1.GraphPane;
            pane.CurveList.Clear();
            
            // título do Gráfico
            pane.Title.Text = $"Validação: {_featureNames[selectedFeatureIndex]}";
            pane.YAxis.Title.Text = $"Valor: {_featureNames[selectedFeatureIndex]}";

            PointPairList listReal = new PointPairList();
            PointPairList listPred = new PointPairList();

            // _lastStartIndex = onde iniciou os dados de leitura
            // começa logo após os 512 passos 
            int startOfPredictionInCsv = _lastStartIndex + SeqLen;

            // plotar APENAS o tamanho da predição 
            // assim as duas linhas ficam exatamente uma sobre a outra
            int len = PredLen; 
            
            // dimensões do tensor de saída
            int outFeats = 7; 
            
            // proteção de índice se o output tiver menos features
            int tensorFeatIndex = (selectedFeatureIndex < outFeats) ? selectedFeatureIndex : 0;
            
            // data de corte para visualização com base nos gráficos de treino
            DateTime targetDate = new DateTime(2024, 10, 6);

            if (viewMode == 0)
            {
                pane.XAxis.Type = AxisType.Linear; // eixo linear simples (0, 1, 2...)
                pane.XAxis.Title.Text = "Horizonte de Previsão (Steps)";
                pane.XAxis.Scale.Format = "0";
                
                for (int i = 0; i < len; i++)
                {
                    // EIXO X
                    double x = i;

                    // DADO REAL (CSV)
                    int realIdx = startOfPredictionInCsv + i;
                    if (realIdx < _allData.Count)
                    {
                        double yReal = _allData[realIdx].X[selectedFeatureIndex];
                        listReal.Add(x, yReal);
                    }

                    // PREDIÇÃO IA
                    float normalizedVal = _lastOutputTensor[0, i, tensorFeatIndex];
                    // desnormalizar
                    double yPred = (normalizedVal * _stdX[selectedFeatureIndex]) + _meanX[selectedFeatureIndex];

                    listPred.Add(x, yPred);
                }
                
                LineItem curvePred = pane.AddCurve("Predição IA", listPred, Color.Orange, SymbolType.None);
                curvePred.Line.Width = 2.5f;
                curvePred.Line.Style = System.Drawing.Drawing2D.DashStyle.Solid; 

                // adicionar curvas
                LineItem curveReal = pane.AddCurve("Real", listReal, Color.Blue, SymbolType.None);
                curveReal.Line.Width = 2.5f;
            
                // ajusta escala
                zedGraphControl1.AxisChange();
                zedGraphControl1.Invalidate();
            }
            else 
            {
                pane.XAxis.Type = AxisType.Date;
                pane.XAxis.Title.Text = "Tempo";
                pane.XAxis.Scale.Format = "yyyy/dd/MM\nHH:mm";

                // 1. DADOS REAIS (AZUL) - FILTRADOS PELA DATA
                // só adiciona ao gráfico se a data for >= 2024-10-11
                for (int i = 0; i < _allData.Count; i++)
                {
                    if (_allData[i].Date < targetDate) continue; // Pula dados antigos

                    double xDate = (double)new XDate(_allData[i].Date);
                    double yReal = _allData[i].X[selectedFeatureIndex];
                    listReal.Add(xDate, yReal);
                }

                // 2. DADOS PREDITOS (LARANJA)
                // A lógica do _lastStartIndex já garante que a predição começa na data certa
                int currentDataIndex = _lastStartIndex + SeqLen;

                foreach (float[] batchData in _predictionHistory)
                {
                    for (int step = 0; step < PredLen; step++)
                    {
                        if (currentDataIndex >= _allData.Count) break;

                        // Não precisamos filtrar aqui pois a inferência já começou na data certa,
                        // mas por segurança, usamos a mesma data real para o eixo X.
                        double xDate = (double)new XDate(_allData[currentDataIndex].Date);
                
                        int flatIndex = (step * outFeats) + tensorFeatIndex;
                        float normalizedVal = batchData[flatIndex];
                        double yPred = (normalizedVal * _stdX[selectedFeatureIndex]) + _meanX[selectedFeatureIndex];

                        listPred.Add(xDate, yPred);
                        currentDataIndex++;
                    }
                }
                LineItem curvePred = pane.AddCurve("Predição IA", listPred, Color.Firebrick, SymbolType.None);
                curvePred.Line.Width = 1.8f;
                curvePred.Line.Style = System.Drawing.Drawing2D.DashStyle.Solid; 

                // adicionar Curvas
                LineItem curveReal = pane.AddCurve("Real", listReal, Color.CornflowerBlue, SymbolType.None);
                curveReal.Line.Width = 1.8f;
            
                // ajusta escala
                zedGraphControl1.AxisChange();
                zedGraphControl1.Invalidate();
            }
            
        }
    }
}