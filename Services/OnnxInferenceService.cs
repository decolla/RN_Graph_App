using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;
using RN_Graph_App.Data;
using RN_Graph_App.Models;

namespace WinFormsOnnxApp
{
    public class OnnxInferenceService
    {
        private int _seqLen;
        private int _featX;
        private int _featMark;
        private int _featAux;

        public OnnxInferenceService(int seqLen, int featX, int featMark, int featAux)
        {
            _seqLen = seqLen;
            _featX = featX;
            _featMark = featMark;
            _featAux = featAux;
        }

        public List<(int Index, float[] Result)> RunInference(string onnxPath, List<TimeSeriesPoint> allData, FeatureScaler scaler, int startIndex, int totalCount, int predLen, int step, Action<int> statusCallback)
        {
            var predictionHistory = new List<(int, float[])>();

            // sessão simples para CPU
            var options = new SessionOptions();
            // usa todos os núcleos disponíveis
            options.InterOpNumThreads = Environment.ProcessorCount;

            using (var session = new InferenceSession(onnxPath, options))
            {
                // limite do loop 
                int maxIndex = totalCount - _seqLen - predLen + 1;

                // percorre os dados em janelas incrementando por 'step'
                for (int i = startIndex; i <= maxIndex; i += step)
                {
                    // prepara dados para essa janela que percorrerá o input
                    var inputWindow = allData.GetRange(i, _seqLen);

                    // passagem de dados pela rede neural
                    var inputs = PrepareInputs(inputWindow, scaler);

                    using (var results = session.Run(inputs))
                    {
                        // armazena os dados de output em memória 
                        var outputRaw = results.First(x => x.Name == "output").AsTensor<float>();
                        float[] batchResult = outputRaw.ToArray();

                        lock (predictionHistory)
                        {
                            // salva o i juntamente ao resultado desse lote na lista
                            predictionHistory.Add((i, batchResult));
                        }
                    }

                    // Callback para informar progresso (opcional)
                    statusCallback?.Invoke(predictionHistory.Count);
                }
            }
            return predictionHistory;
        }

        // função auxiliar para preparar os tensores
        private List<NamedOnnxValue> PrepareInputs(List<TimeSeriesPoint> window, FeatureScaler scaler)
        {
            // prepara os tensores
            var tensorX = new DenseTensor<float>(new[] { 1, _seqLen, _featX });
            var tensorMark = new DenseTensor<float>(new[] { 1, _seqLen, _featMark });
            var tensorAux = new DenseTensor<float>(new[] { 1, _seqLen, _featAux });

            for (int t = 0; t < _seqLen; t++)
            {
                var row = window[t];

                // normalizar batch_x
                for (int f = 0; f < _featX; f++)
                    // subtrai a mediana e dividimos pelo IQR
                    tensorX[0, t, f] = (float)((row.X[f] - scaler.MedianX[f]) / scaler.IqrX[f]);

                // normalizar o temporal
                tensorMark[0, t, 0] = (float)((row.Date.Month - 1) / 11.0 - 0.5);
                tensorMark[0, t, 1] = (float)((row.Date.Day - 1) / 30.0 - 0.5);
                tensorMark[0, t, 2] = (float)((int)row.Date.DayOfWeek / 6.0 - 0.5);
                tensorMark[0, t, 3] = (float)(row.Date.Hour / 23.0 - 0.5);
                tensorMark[0, t, 4] = (float)(row.Date.Minute / 59.0 - 0.5);

                // normalizar aux (passagem direta de valor bruto)
                for (int f = 0; f < _featAux; f++) // tensorAux[0, t, f] = (float)row.Aux[f];
                    tensorAux[0, t, f] = (float)((row.Aux[f] - scaler.MedianAux[f]) / scaler.IqrAux[f]);
            }

            // cira pacote com os tensores criados, com os nomes esperados
            return new List<NamedOnnxValue>
            {
                NamedOnnxValue.CreateFromTensor("batch_x", tensorX),
                NamedOnnxValue.CreateFromTensor("batch_x_mark", tensorMark),
                NamedOnnxValue.CreateFromTensor("batch_x_aux", tensorAux)
            };
        }
    }
}