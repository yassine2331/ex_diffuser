import pandas as pd
import os

# Define the file path
CSV_FILE = "master_evaluation_results.csv"

def generate_latex_from_file(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found. Please ensure the CSV file is in the same directory.")
        return

    # Load the CSV
    df = pd.read_csv(file_path)
    
    # Clean column names
    df.columns = df.columns.str.strip()

    # Define metric behavior for bolding
    # Lower is better (down arrow)
    lower_better = ['L1', 'L1.5', 'L2', 'sFID', 'Face Cosine']
    # Higher is better (up arrow)
    higher_better = ['Flip (Ext)', 'Face ID %']

    # Group by Task and Direction to create separate tables
    grouped = df.groupby(['Task', 'Dir'])

    for (task, direction), group in grouped:
        print(f"\n% --- Table for {task} ({direction}) ---")
        
        # LaTeX Table Setup
        # Columns: Model | L1 | L1.5 | L2 | Flip(Ext) | Face ID% | Face Cos | sFID
        latex_lines = [
            "\\begin{table}[h]",
            "\\centering",
            "\\small",
            "\\begin{tabular}{|l|ccc|c|cc|c|}",
            "\\hline",
            "\\multicolumn{1}{|c|}{} & \\multicolumn{3}{|c|}{\\textbf{Closeness}} & \\multicolumn{1}{|c|}{\\textbf{Validity}} & \\multicolumn{2}{|c|}{\\textbf{Identity}} & \\multicolumn{1}{|c|}{\\textbf{Realism}} \\\\",
            "\\hline",
            "\\textbf{Metric} & $l_1 \\downarrow$ & $l_{1.5} \\downarrow$ & $l_2 \\downarrow$ & Flip (Ext) $\\uparrow$ & ID \\% $\\uparrow$ & Cosine $\\downarrow$ & sFID $\\downarrow$ \\\\",
            "\\hline"
        ]

        # Process each model in the group
        for _, row in group.iterrows():
            model_name = str(row['Model']).replace('_', '\\_')
            
            formatted_vals = {}
            for col in lower_better + higher_better:
                if col not in df.columns:
                    formatted_vals[col] = "-"
                    continue
                
                val = row[col]
                # Bolding logic based on group context
                if col in lower_better:
                    is_best = (val == group[col].min())
                else:
                    is_best = (val == group[col].max())
                
                # Formatting decimals (3 places for small numbers, 2 for FID/Percentages)
                if col in ['sFID', 'Face ID %']:
                    str_val = f"{val:.2f}"
                else:
                    str_val = f"{val:.3f}"
                    
                formatted_vals[col] = f"\\textbf{{{str_val}}}" if is_best else str_val

            # Row string construction
            row_str = (
                f"{model_name} & "
                f"{formatted_vals['L1']} & {formatted_vals['L1.5']} & {formatted_vals['L2']} & "
                f"{formatted_vals['Flip (Ext)']} & "
                f"{formatted_vals['Face ID %']} & {formatted_vals['Face Cosine']} & "
                f"{formatted_vals['sFID']} \\\\"
            )
            latex_lines.append(row_str)
        
        # Footer
        latex_lines.extend([
            "\\hline",
            "\\end{tabular}",
            f"\\caption{{Evaluation results for {task} task ({direction}).}}",
            "\\end{table}"
        ])
        
        print("\n".join(latex_lines))

if __name__ == "__main__":
    generate_latex_from_file(CSV_FILE)