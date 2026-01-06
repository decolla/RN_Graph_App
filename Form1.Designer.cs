namespace RN_Graph_App;

partial class Form1
{
    /// <summary>
    ///  Required designer variable.
    /// </summary>
    private System.ComponentModel.IContainer components = null;

    /// <summary>
    ///  Clean up any resources being used.
    /// </summary>
    /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
    protected override void Dispose(bool disposing)
    {
        if (disposing && (components != null))
        {
            components.Dispose();
        }

        base.Dispose(disposing);
    }

    #region Windows Form Designer generated code

    /// <summary>
    /// Required method for Designer support - do not modify
    /// the contents of this method with the code editor.
    /// </summary>
    private void InitializeComponent()
    {
        components = new System.ComponentModel.Container();
        button1 = new System.Windows.Forms.Button();
        button2 = new System.Windows.Forms.Button();
        button3 = new System.Windows.Forms.Button();
        panel1 = new System.Windows.Forms.Panel();
        zedGraphControl1 = new ZedGraph.ZedGraphControl();
        comboBox1 = new System.Windows.Forms.ComboBox();
        panel1.SuspendLayout();
        SuspendLayout();
        // 
        // button1
        // 
        button1.Location = new System.Drawing.Point(12, 12);
        button1.Name = "button1";
        button1.Size = new System.Drawing.Size(130, 66);
        button1.TabIndex = 0;
        button1.Text = "Carregar Arquivo CSV";
        button1.UseVisualStyleBackColor = true;
        button1.Click += button1_Click;
        // 
        // button2
        // 
        button2.Location = new System.Drawing.Point(148, 12);
        button2.Name = "button2";
        button2.Size = new System.Drawing.Size(130, 66);
        button2.TabIndex = 1;
        button2.Text = "Carregar IA (ONNX)";
        button2.UseVisualStyleBackColor = true;
        button2.Click += button2_Click;
        // 
        // button3
        // 
        button3.Location = new System.Drawing.Point(413, 12);
        button3.Name = "button3";
        button3.Size = new System.Drawing.Size(130, 66);
        button3.TabIndex = 2;
        button3.Text = "Processar";
        button3.UseVisualStyleBackColor = true;
        button3.Click += button3_Click;
        // 
        // panel1
        // 
        panel1.Controls.Add(zedGraphControl1);
        panel1.Location = new System.Drawing.Point(12, 84);
        panel1.Name = "panel1";
        panel1.Size = new System.Drawing.Size(776, 354);
        panel1.TabIndex = 3;
        // 
        // zedGraphControl1
        // 
        zedGraphControl1.Location = new System.Drawing.Point(0, 0);
        zedGraphControl1.Margin = new System.Windows.Forms.Padding(4, 3, 4, 3);
        zedGraphControl1.Name = "zedGraphControl1";
        zedGraphControl1.ScrollGrace = 0D;
        zedGraphControl1.ScrollMaxX = 0D;
        zedGraphControl1.ScrollMaxY = 0D;
        zedGraphControl1.ScrollMaxY2 = 0D;
        zedGraphControl1.ScrollMinX = 0D;
        zedGraphControl1.ScrollMinY = 0D;
        zedGraphControl1.ScrollMinY2 = 0D;
        zedGraphControl1.Size = new System.Drawing.Size(776, 354);
        zedGraphControl1.TabIndex = 0;
        zedGraphControl1.UseExtendedPrintDialog = true;
        // 
        // comboBox1
        // 
        comboBox1.FormattingEnabled = true;
        comboBox1.Location = new System.Drawing.Point(286, 35);
        comboBox1.Name = "comboBox1";
        comboBox1.Size = new System.Drawing.Size(121, 23);
        comboBox1.TabIndex = 4;
        comboBox1.DropDownStyle = ComboBoxStyle.DropDownList;
        comboBox1.Items.AddRange(new string[] {
            "PEHIST",
            "PSHIST",
            "REGULADOR1",
            "REGULADOR2",
            "PDT1",
            "PDT2",
            "FT1"
        });
        comboBox1.SelectedIndex = 0;
        // 
        // Form1
        // 
        AccessibleRole = System.Windows.Forms.AccessibleRole.None;
        AutoScaleDimensions = new System.Drawing.SizeF(7F, 15F);
        AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
        BackColor = System.Drawing.SystemColors.Control;
        ClientSize = new System.Drawing.Size(800, 450);
        Controls.Add(comboBox1);
        Controls.Add(panel1);
        Controls.Add(button3);
        Controls.Add(button2);
        Controls.Add(button1);
        ForeColor = System.Drawing.SystemColors.ActiveCaptionText;
        Text = "Form1";
        TopMost = true;
        panel1.ResumeLayout(false);
        ResumeLayout(false);
    }

    private System.Windows.Forms.ComboBox comboBox1;

    private ZedGraph.ZedGraphControl zedGraphControl1;

    private System.Windows.Forms.Button button2;
    private System.Windows.Forms.Button button3;
    private System.Windows.Forms.Panel panel1;

    private System.Windows.Forms.Button button1;

    #endregion
}