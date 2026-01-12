using System;
using System.Windows.Forms;

namespace WinFormsOnnxApp
{
    static class ProgramV2
    {
        /// <summary>
        /// Ponto de entrada principal para o aplicativo.
        /// </summary>
        [STAThread]
        static void Main()
        {
            Application.EnableVisualStyles();
            Application.SetCompatibleTextRenderingDefault(false);
            Application.Run(new Graph_RN_FormV2());
        }
    }
}