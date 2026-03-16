import pandas as pd
import matplotlib.pyplot as plt
import re
import os

# Define the file path
CSV_FILE = "master_evaluation_results.csv"

def generate_sfid_tradeoff_plots(file_path):
    if not os.path.exists(file_path):
        print(f"Error: {file_path} not found.")
        return

    # Load and clean the CSV
    df = pd.read_csv(file_path)
    df.columns = df.columns.str.strip()

    # Parse model info
    def parse_model_info(name):
        method = "MyModel" if "MyModel" in name else "Baseline"
        match = re.search(r'_(\d+)$', name)
        param = int(match.group(1)) if match else 0
        return method, param

    df[['Method', 'Param']] = df['Model'].apply(lambda x: pd.Series(parse_model_info(x)))

    # Group by Task and Direction
    groups = df.groupby(['Task', 'Dir'])

    for (task, direction), group in groups:
        plt.figure(figsize=(8, 6))
        
        for method, color in [("MyModel", "green"), ("Baseline", "red")]:
            m_data = group[group['Method'] == method].sort_values('Param')
            
            if m_data.empty:
                continue

            # Plot line and crosses (X=sFID, Y=Flip Rate)
            plt.plot(m_data['sFID'], m_data['Flip (Ext)'], 
                     marker='x', markersize=8, color=color, 
                     linestyle='-', linewidth=1.5, label=method)
            
            # Annotate parameter values (10, 20, 50)
            for _, row in m_data.iterrows():
                plt.text(row['sFID'], row['Flip (Ext)'], f"{row['Param']}", 
                         fontsize=9, verticalalignment='bottom', horizontalalignment='left')

        # Formatting
        plt.xlabel('sFID Score (Realism) $\\downarrow$')
        plt.ylabel('Flip Rate (External) (Validity) $\\uparrow$')
        plt.title(f'Realism-Validity Trade-off: {task} ({direction})')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Flip the y-axis as requested (higher flip rate at the bottom)
        plt.gca().invert_yaxis()
        
        # Save plot
        clean_dir = direction.replace(' -> ', '_')
        filename = f"plot_sfid_{task.lower()}_{clean_dir}_inverted.png"
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Generated sFID plot: {filename}")
        plt.close()

if __name__ == "__main__":
    generate_sfid_tradeoff_plots(CSV_FILE)