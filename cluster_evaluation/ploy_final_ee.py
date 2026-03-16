import matplotlib.pyplot as plt
import time

# 1. Start Timing the "Data Read"
start_time = time.time()


# Smiling,0 -> 1,Baseline_final_2,0.9475,0.3203,10.51,0,0,0.0193,0.0335,0.0259
# Smiling,0 -> 1,Baseline_final,0.9984,0.6217,11.65,0,0,0.0227,0.0428,0.0319
# Smiling,0 -> 1,Baseline_final_3,1.0,0.9831,24.27,0,0,0.0417,0.0779,0.0588
# Smiling,0 -> 1,Baseline_final_1,0.6186,0.1003,9.78,0,0,0.0166,0.0268,0.0215


# Simulating reading from a DataFrame (Data transcribed from your images)
data = [
    {"Model": "Baseline_final_0.3",    "Flip (Ext)": 0.6186, "sFID": 9.78,  "Face Cosine": 0, "L1": 0.0166, "L2": 0.0268, "L1.5": 0.0215},
    {"Model": "Baseline_final_0.4",    "Flip (Ext)": 0.9475, "sFID": 10.51, "Face Cosine": 0, "L1": 0.0193, "L2": 0.0335, "L1.5": 0.0259},
    {"Model": "Baseline_final_0.5",      "Flip (Ext)": 0.9984, "sFID": 11.65, "Face Cosine": 0, "L1": 0.0227, "L2": 0.0428, "L1.5": 0.0319},
    {"Model": "Baseline_final_0.7",    "Flip (Ext)": 1.0, "sFID": 15.67,  "Face Cosine": 0, "L1": 0.0313, "L2": 0.0617,"L1.5": 0.0452},
    {"Model": "Baseline_final_1",    "Flip (Ext)": 1.0,    "sFID": 24.27, "Face Cosine": 0, "L1": 0.0417, "L2": 0.0779, "L1.5": 0.0588},
    
    #Smiling,0 -> 1,Baseline_final_0.7 ,1.0,0.9302,15.67,0,0,0.0313,0.0617,0.0452
]
# this is with s = 5 
# Avg	0 -> 1	MyModel_CBM_final_1	0.8417	0.3113	10.66	0	0	0.0137	0.0336	0.0231
# Avg	0 -> 1	MyModel_CBM_final_5	0.9969	0.7745	12.31	0	0	0.017	0.0425	0.0292
# Avg	0 -> 1	MyModel_CBM_final_3	0.9884	0.663	11.48	0	0	0.0162	0.0404	0.0277
# Avg	0 -> 1	MyModel_CBM_final_2	0.7703	0.1416	10.62	0	0	0.0136	0.0335	0.0231
# Avg	0 -> 1	MyModel_CBM_final_4	0.9667	0.5516	11.38	0	0	0.0153	0.038	0.0261
#{"Model": "MyModel_CBM_final_1", "Flip (Ext)": 0.9969, "sFID": 12.31, "Face Cosine": 0, "L1": 0.0170, "L2": 0.0425},
# data.extend([
#     {"Model": "MyModel_CBM_final_1_s5", "Flip (Ext)": 0.8417, "sFID": 10.66, "Face Cosine": 0, "L1": 0.0137, "L2": 0.0336}, # file in final 
#     {"Model": "MyModel_CBM_final_5_s5", "Flip (Ext)": 0.9969, "sFID": 12.31, "Face Cosine": 0, "L1": 0.017, "L2": 0.0425}, # file in fina_1
#     {"Model": "MyModel_CBM_final_3_s5", "Flip (Ext)": 0.9884, "sFID": 11.48, "Face Cosine": 0, "L1": 0.0162, "L2": 0.0404},
#     #{"Model": "MyModel_CBM_final_2_s5", "Flip (Ext)": 0.7703, "sFID": 10.62, "Face Cosine": 0, "L1": 0.0136, "L2": 0.0335},
#     {"Model": "MyModel_CBM_final_2_s5", "Flip (Ext)": 0.9667, "sFID": 11.38, "Face Cosine": 0, "L1": 0.0153, "L2": 0.038} # file in "MyModel_CBM_final_4_s5"
# ])


#this is with s = 4
# Smiling 0 -> 1 MyModel_CBM_final_5      0.9917      0.7055 11.99          0            0 0.0165 0.0414 0.0283
# Smiling 0 -> 1 MyModel_CBM_final_4      0.9898      0.6803 11.54          0            0 0.0161 0.0403 0.0276
# Smiling 0 -> 1 MyModel_CBM_final_3      0.9786      0.5916 11.42          0            0 0.0156 0.0389 0.0267
# Smiling 0 -> 1 MyModel_CBM_final_2      0.9486      0.4516 11.29          0            0 0.0147 0.0366 0.0251
# Smiling 0 -> 1 MyModel_CBM_final_1      0.7806      0.2258 10.63          0            0 0.0132 0.0325 0.0224
data.extend([
    {"Model": "MyModel_CBM_final_5_s4", "Flip (Ext)": 0.9917, "sFID": 11.99, "Face Cosine": 0, "L1": 0.0165, "L2": 0.0414, "L1.5": 0.0283},
    {"Model": "MyModel_CBM_final_4_s4", "Flip (Ext)": 0.9898, "sFID": 11.54, "Face Cosine": 0, "L1": 0.0161, "L2": 0.0403, "L1.5": 0.0276},
    {"Model": "MyModel_CBM_final_3_s4", "Flip (Ext)": 0.9786, "sFID": 11.42, "Face Cosine": 0, "L1": 0.0156, "L2": 0.0389, "L1.5": 0.0267},
    {"Model": "MyModel_CBM_final_2_s4", "Flip (Ext)": 0.9486, "sFID": 11.29, "Face Cosine": 0, "L1": 0.0147, "L2": 0.0366, "L1.5": 0.0251},
    {"Model": "MyModel_CBM_final_1_s4", "Flip (Ext)": 0.7806, "sFID": 10.63, "Face Cosine": 0, "L1": 0.0132, "L2": 0.0325, "L1.5": 0.0224} 
])

# this is with s = 3
# Smiling 0 -> 1 MyModel_CBM_final_5      0.9873      0.6002 11.63          0            0 0.0161 0.0402 0.0276
# Smiling 0 -> 1 MyModel_CBM_final_4      0.9809      0.5458 11.17          0            0 0.0157 0.0393 0.0269
# Smiling 0 -> 1 MyModel_CBM_final_3      0.9619      0.4558 11.11          0            0 0.0152 0.0378 0.0259
# Smiling 0 -> 1 MyModel_CBM_final_2      0.9005      0.3258 10.93          0            0 0.0142 0.0352 0.0242
# Smiling 0 -> 1 MyModel_CBM_final_1      0.6828      0.1392 10.43          0            0 0.0126 0.0309 0.0213
# data.extend([
#     {"Model": "MyModel_CBM_final_5_s3", "Flip (Ext)": 0.9873, "sFID": 11.63, "Face Cosine": 0, "L1": 0.0161, "L2": 0.0402, "L1.5": 0.0276},
#     {"Model": "MyModel_CBM_final_4_s3", "Flip (Ext)": 0.9809, "sFID": 11.17, "Face Cosine": 0, "L1": 0.0157, "L2": 0.0393, "L1.5": 0.0269},
#     {"Model": "MyModel_CBM_final_3_s3", "Flip (Ext)": 0.9619, "sFID": 11.11, "Face Cosine": 0, "L1": 0.0152, "L2": 0.0378, "L1.5": 0.0259},
#     {"Model": "MyModel_CBM_final_2_s3", "Flip (Ext)": 0.9005, "sFID": 10.93, "Face Cosine": 0, "L1": 0.0142, "L2": 0.0352, "L1.5": 0.0242},
#     {"Model": "MyModel_CBM_final_1_s3", "Flip (Ext)": 0.6828, "sFID": 10.43, "Face Cosine": 0, "L1": 0.0126, "L2": 0.0309, "L1.5": 0.0213} 
# ])
end_time = time.time()
print(f"Data read time: {end_time - start_time:.6f} seconds")

# 2. Define the Function
def plot_metric_vs_flip(metric_name, reverse_y=True):
    """
    Plots a chosen metric (X-axis) against Flip Rate (Y-axis).
    Args:
        metric_name (str): The key from the data dict (e.g., 'L1', 'L2', 'sFID').
        reverse_y (bool): If True, inverts Y-axis so 1.0 is at the bottom (or top if preferred).
    """
    
    # Extract data
    my_model_s5_x = []
    my_model_s5_y = []
    my_model_s4_x = []
    my_model_s4_y = []
    my_model_s3_x = []
    my_model_s3_y = []
    baseline_x = []
    baseline_y = []
    
    # Sort data into lists based on model type
    # writing all models using sn to avoid typos and ensure consistency
    
    for row in data:
        x_val = row[metric_name]
        y_val = row["Flip (Ext)"]
        
        if "MyModel" in row["Model"]:
            if "_s5" in row["Model"]:
                my_model_s5_x.append(x_val)
                my_model_s5_y.append(y_val)
            elif "_s4" in row["Model"]:
                my_model_s4_x.append(x_val)
                my_model_s4_y.append(y_val)
            elif "_s3" in row["Model"]:
                my_model_s3_x.append(x_val)
                my_model_s3_y.append(y_val)
        else:
            baseline_x.append(x_val)
            baseline_y.append(y_val)

    plt.figure(figsize=(14, 10))
    # increasing hte label size 

    # Plot points
    #plt.scatter(my_model_s5_x, my_model_s5_y, color='green', s=100, label='MyModel S5', edgecolors='k', zorder=3)
    plt.scatter(my_model_s4_x, my_model_s4_y, color='blue', s=100, label='C-VCE (Ours)', edgecolors='k', zorder=3)
    #plt.scatter(my_model_s3_x, my_model_s3_y, color='purple', s=100, label='MyModel S3', edgecolors='k', zorder=3)
    plt.scatter(baseline_x, baseline_y, color='red', marker='s', s=100, label='L-DVCE (Baseline)', edgecolors='k', zorder=3)

    # Labels and Title
    plt.xlabel(f"{metric_name} (Lower is better)", fontsize=36)
    plt.ylabel("Flip Ext (1.0 is best)", fontsize=36)
    plt.title(f"Trade-off: {metric_name} vs. Flip Rate", fontsize=36)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend(fontsize=30)

    #making the number biggger 

    plt.xticks(fontsize=30)
    plt.yticks(fontsize=30)

    # drawing lines to connect the points of the same model
    #plt.plot(my_model_s5_x, my_model_s5_y, color='green', linestyle='--', zorder=2)
    plt.plot(my_model_s4_x, my_model_s4_y, color='blue', linestyle='--', zorder=2)
    #plt.plot(my_model_s3_x, my_model_s3_y, color='purple', linestyle='--', zorder=2)
    plt.plot(baseline_x, baseline_y, color='red', linestyle='--', zorder=2)

    # Handle Axis Reversal (User Request)
    # If we want 'Flip Ext Reversed', we invert Y so 1.0 is at the bottom 
    # and 0.0 is at the top (or vice versa depending on your specific visual preference).
    if reverse_y:
        plt.gca().invert_yaxis()
        plt.title(f"Trade-off: {metric_name} vs. Flip Rate (Reversed Axis)", fontsize=36)

    plt.savefig(f"{metric_name}_vs_Flip_Ext.png", dpi=300)
    

# 3. Call the function
# You can change 'L1' to 'L2', 'sFID', or 'Face Cosine'
plot_metric_vs_flip('L1')
plot_metric_vs_flip('L1.5')
plot_metric_vs_flip('L2')
plot_metric_vs_flip('sFID')




# Baseline - Verified: 2.3125
# CBM - Verified: 2.2805
# Baseline - Verified: 0.9508634868421053
# CBM - Verified: 0.9446975973487987
# Baseline - Errors: 723
# CBM - Errors: 690
# Baseline : 4864
# CBM  4828
# Baseline - Average Distance/Similarity: 0.3884285553042764
# CBM - Average Distance/Similarity: 0.41986814188069543
# ##########################################################
# Baseline - Failed (No Face Detected): 187
# CBM - Failed (No Face Detected): 118
# Baseline - Total Processed: 4864
# CBM - Total Processed: 4828