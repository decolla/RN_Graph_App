using System.Globalization;
using RN_Graph_App.Models;

namespace WinFormsOnnxApp
{
    public static class CsvDataLoader
    {
        public static List<TimeSeriesPoint> Load(string path, int featX, int featAux)
        {
            // le todas as linhas do arquivo
            var lines = File.ReadAllLines(path);
            // armazena os dados em uma lista
            var allData = new List<TimeSeriesPoint>();

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
                row.X = new float[featX];
                for (int j = 0; j < featX; j++)
                    row.X[j] = ParseFloat(parts[j + 1]);

                // preenche o vetor Aux (batch_aux)
                row.Aux = new float[featAux];
                for (int j = 0; j < featAux; j++)
                    row.Aux[j] = ParseFloat(parts[j + 1 + featX]);

                // adicioNa os dados na lista
                allData.Add(row);
            }
            return allData;
        }

        // formatação de valores para float 
        private static float ParseFloat(string val)
        {
            if (float.TryParse(val, NumberStyles.Any, CultureInfo.InvariantCulture, out float res)) return res;
            return 0f;
        }
    }
}