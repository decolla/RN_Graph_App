namespace RN_Graph_App.Models;

public class TimeSeriesPoint
{
    public DateTime Date { get; set; }
    public float[] X { get; set; }
    public float[] Aux { get; set; }
}