using System.Globalization;
using Microsoft.ML.OnnxRuntime.Tensors;
using RN_Graph_App.Data;
using RN_Graph_App.Models;
using ZedGraph;

namespace WinFormsOnnxApp
{
    public partial class Graph_RN_FormV2 : Form
    {
        // CONFIGURAÇÕES DO MODELO
        private const int SeqLen = 720; // janela de imput
        private const int PredLen = 336; // janela da predição

        // definição das features 
        private const int FeatX = 7;

        private readonly string[] _featureNames = new string[]
        {
            "PEHIST", "PSHIST", "REGULADOR1", "REGULADOR2", "PDT1", "PDT2", "FT1"
        };

        // path para os arquivos
        private string _csvPath;
        private string _onnxPath = "final_model.onnx"; // ultimo modelo mandado como base

        // Estado dos dados
        private List<TimeSeriesPoint> _allData;
        private Tensor<float> _lastOutputTensor;
        private int _lastStartIndex = -1;

        // VARIÁVEL PARA TODOS OS LOTES: lista para armazenar os blocos de predição (indíce original, e resultado)
        private List<(int Index, float[] Result)> _predictionHistory;

        // MÓDULOS (Serviços)
        private FeatureScaler _scaler;
        private OnnxInferenceService _onnxService;

        public Graph_RN_FormV2()
        {
            InitializeComponent();
            SetupGraph();

            // Inicializa os módulos
            _scaler = new FeatureScaler(FeatX, 4); // 4 = FeatAux
            _onnxService = new OnnxInferenceService(SeqLen, FeatX, 5, 4); // 5 = FeatMark, 4 = FeatAux

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
            myPane.Title.Text = "Validação: real e predição";
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
                // Delega a leitura para a classe CsvDataLoader
                _allData = CsvDataLoader.Load(path, FeatX, 4); // 4 = FeatAux

                // verifica se o arquivo csv tem tamanho suficiente
                if (_allData.Count < SeqLen + PredLen)
                {
                    MessageBox.Show($"CSV insuficiente. Mínimo: {SeqLen + PredLen} linhas.");
                    return;
                }

                // CALCULA A MEDIANA E IQR (Delega para FeatureScaler)
                _scaler.FitScalers(_allData, _featureNames);

                lblStatus.Text = $"Dados carregados: {_allData.Count} linhas.";
                btnRunInference.Enabled = true;
            }
            catch (Exception ex)
            {
                MessageBox.Show("Erro ao ler CSV: " + ex.Message);
            }
        }

        private async void btnRunInference_Click(object sender, EventArgs e)
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


                // ========================================================================
                // AJUSTE DE VELOCIDADE (12 = EQUILÍBRIO, 24 = RÁPIDO, 48 = MUITO RÁPIDO)
                int step = 24;
                // ========================================================================


                // Backtesting: separa os últimos x dados reais (predição)
                int totalCount = _allData.Count;
                int numTest = (int)(totalCount * 0.1); // 10% para teste

                // fórmula: len(df) - num_test - seq_len
                _lastStartIndex = totalCount - numTest - SeqLen;
                // indice onde começa a leitura dos 512 dados de entrada

                if (_lastStartIndex < 0)
                {
                    MessageBox.Show("Dados insuficientes.");
                    return;
                }

                // inicia a lista de histórico
                _predictionHistory = new List<(int, float[])>();

                // Configura um callback para atualizar o status na UI durante o processamento
                // Usando Invoke para garantir thread-safety com a UI
                Action<int> progressCallback = (count) =>
                {
                    // Opcional: Atualizar label aqui se necessário
                };

                // task para não travar a tela enquanto roda 
                await Task.Run(() =>
                {
                    // Executa a inferência usando o serviço dedicado
                    _predictionHistory = _onnxService.RunInference(
                        _onnxPath,
                        _allData,
                        _scaler,
                        _lastStartIndex,
                        totalCount,
                        PredLen,
                        step,
                        progressCallback
                    );
                });

                // armazena o primeiro tensor para modo de vizualização estático
                if (_predictionHistory.Count > 0)
                {
                    // [0] acessa o array da tupla
                    _lastOutputTensor = new DenseTensor<float>(_predictionHistory[0].Result,
                        new ReadOnlySpan<int>(new[] { 1, PredLen, 7 })); // 7 é o Featx
                }

                cmbColumns.Enabled = true;
                cmbViewMode.Enabled = true;
                // desenha as linhas no gráfico
                UpdatePlot();
                lblStatus.Text = $"Gráfico gerado com {_predictionHistory.Count} pontos. Selecione o modo de visualização.";
            }
            catch (Exception ex)
            {
                MessageBox.Show($"Erro na inferência: {ex.Message}");
            }
        }

        // comboBox para trocar de gráfico de acordo com a coluna 
        private void cmbColumns_SelectedIndexChanged(object sender, EventArgs e)
        {
            if (_lastOutputTensor != null)
            {
                UpdatePlot();
            }
        }

        // forma o gráfico de real vs predição
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

            // reseta o zoom
            zedGraphControl1.ZoomOutAll(pane);
            pane.XAxis.Scale.MinAuto = true;
            pane.YAxis.Scale.MinAuto = true;
            pane.YAxis.Scale.MaxAuto = true;
            pane.XAxis.Scale.MaxAuto = true;

            pane.CurveList.Clear();

            // título do gráfico
            pane.Title.Text = $"Validação: {_featureNames[selectedFeatureIndex]}";
            pane.YAxis.Title.Text = $"Valor: {_featureNames[selectedFeatureIndex]}";

            PointPairList listReal = new PointPairList();
            PointPairList listPred = new PointPairList();

            // dimensões do tensor de saída
            int outFeats = 7;

            // proteção de índice se o output tiver menos features
            int tensorFeatIndex = (selectedFeatureIndex < outFeats) ? selectedFeatureIndex : 0;

            // data de corte para visualização com base nos gráficos de treino
            DateTime targetDateStart = new DateTime(2024, 10, 6);
            DateTime targetDateEnd = new DateTime(2024, 12, 13);

            if (viewMode == 0)
            {
                pane.XAxis.Type = AxisType.Linear; // eixo linear simples (0, 1, 2...)
                pane.XAxis.Title.Text = "Horizonte de previsão (Steps)";
                pane.XAxis.Scale.Format = "0";

                // pega o índice real de início da predição no CSV
                int startReal = _predictionHistory[0].Index + SeqLen;

                for (int i = 0; i < PredLen; i++)
                {
                    // EIXO X
                    double x = i;

                    // DADO REAL (CSV)
                    int realIdx = startReal + i;
                    if (realIdx < _allData.Count)
                    {
                        listReal.Add(x, _allData[realIdx].X[selectedFeatureIndex]);
                    }

                    // PREDIÇÃO IA
                    float normalizedVal = _lastOutputTensor[0, i, tensorFeatIndex];
                    // desnormalizar (Usa o Scaler)
                    double yPred = (normalizedVal * _scaler.IqrX[selectedFeatureIndex]) + _scaler.MedianX[selectedFeatureIndex];

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
            else // HISTÓRICO COMPLETO
            {
                pane.XAxis.Type = AxisType.Date;
                pane.XAxis.Title.Text = "Tempo";
                pane.XAxis.Scale.Format = "yyyy/dd/MM\nHH:mm";

                // DADOS REAIS 
                // só adiciona ao gráfico se a data for >= 2024-10-11
                for (int i = 0; i < _allData.Count; i++)
                {
                    if (_allData[i].Date < targetDateStart || _allData[i].Date > targetDateEnd) continue; // pula dados antigos e aplica limite aos novos

                    double xDate = (double)new XDate(_allData[i].Date);
                    listReal.Add(xDate, _allData[i].X[selectedFeatureIndex]);
                }

                // DADOS PREDITOS
                // usa o índice original salvo na lista de histórico
                foreach (var item in _predictionHistory)
                {
                    int originalIndex = item.Index; // onde a janela de imput começou no CSV
                    float[] result = item.Result; // resultado dessa predição

                    int predictionDataIdx = originalIndex + SeqLen; // índice do primeiro ponto predito no CSV

                    if (predictionDataIdx >= _allData.Count) break; // proteção

                    if (_allData[predictionDataIdx].Date < targetDateStart || _allData[predictionDataIdx].Date > targetDateEnd) continue; // pula dados antigos e restringe novos

                    double xDate = (double)new XDate(_allData[predictionDataIdx].Date);

                    // verifica o índice do tensor
                    int flatIndex = (0 * outFeats) + tensorFeatIndex;
                    float normalizedVal = result[flatIndex];
                    // aplica a desnormalização (Usa o Scaler)
                    double yPred = (normalizedVal * _scaler.IqrX[selectedFeatureIndex]) + _scaler.MedianX[selectedFeatureIndex];

                    listPred.Add(xDate, yPred);

                }

                // adicionar Curvas
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