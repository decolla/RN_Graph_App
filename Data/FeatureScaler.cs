using RN_Graph_App.Models;

namespace RN_Graph_App.Data
{
    public class FeatureScaler
    {
        public double[] MedianX { get; private set; }
        public double[] IqrX { get; private set; }
        public double[] MedianAux { get; private set; }
        public double[] IqrAux { get; private set; }

        private int _featX;
        private int _featAux;

        public FeatureScaler(int featX, int featAux)
        {
            _featX = featX;
            _featAux = featAux;
        }

        // função de cálculo de média e variação
        public void FitScalers(List<TimeSeriesPoint> allData, string[] featureNames)
        {
            MedianX = new double[_featX]; // vetor de mediana
            IqrX = new double[_featX]; // vetor de IQR
            MedianAux = new double[_featAux];
            IqrAux = new double[_featAux];

            // define o limite de treino para 70% dos dados
            int trainSize = (int)(allData.Count * 0.7);

            // função local para calcular mediana e IQR
            void CalcRobustStats(List<double> values, out double median, out double iqr)
            {
                // valores de início
                if (values.Count == 0)
                {
                    median = 0;
                    iqr = 1;
                    return;
                }

                values.Sort(); // ordena para achar quartis
                int count = values.Count;

                // mediana
                if (count % 2 == 0)
                    median = (values[count / 2 - 1] + values[count / 2]) / 2.0;
                else
                    median = values[count / 2];

                // IQR = Q3 - Q1
                double q1 = values[(int)(count * 0.25)];
                double q3 = values[(int)(count * 0.75)];

                iqr = q3 - q1;
                if (iqr == 0) iqr = 1; // proteção contra divisão por zero
            }

            // calcula estatísticas para features X
            for (int i = 0; i < _featX; i++)
            {
                // pega todos os valores da coluna 'i' até o índice trainSize
                var colValues = allData.Take(trainSize).Select(r => (double)r.X[i]).ToList();
                CalcRobustStats(colValues, out double median, out double iqr);
                MedianX[i] = median;
                IqrX[i] = iqr;
            }

            // calcula estatísticas para Aux
            for (int i = 0; i < _featAux; i++)
            {
                var colValues = allData.Take(trainSize).Select(r => (double)r.Aux[i]).ToList();
                CalcRobustStats(colValues, out double median, out double iqr);
                MedianAux[i] = median;
                IqrAux[i] = iqr;
            }
        }
    }
}