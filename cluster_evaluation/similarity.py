from deepface import DeepFace
from tqdm import tqdm

PATH_REAL_CBM = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_final_4/Smiling_0_1/images_real"    # Real CelebA images
#PATH_FAKE = "./fid_generated_images_ddpm2"            # Your generated images
PATH_CBM =      "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_final_4/Smiling_0_1/images_cbm"  # --- IGNORE ---

PATH_REAL_BASELINE = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_final/Smiling_0_1/images_real"    # Real CelebA images
PATH_BASELINE = "/home/oueslatiy/MasterThesis/ex_diffuser/cluster_evaluation/Smiling_final/Smiling_0_1/images_baseline"  # --- IGNORE ---

metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]


result_cbm = []
result_baseline = []
same_face_cbm = 0
same_face_baseline = 0
#iterate ovel all the images in the folder and compare them
errors_cbm = 0
errors_baseline = 0
fail_cbm = 0
fail_baseline = 0
total_cbm = 0
total_baseline = 0
for i in tqdm(range(1000,6400)):
    real_image_path_baseline = f"{PATH_REAL_BASELINE}/image_{i}.png"
    real_image_path_cbm = f"{PATH_REAL_CBM}/image_{i}.png"
    baseline_image_path = f"{PATH_BASELINE}/image_{i}.png"
    cbm_image_path = f"{PATH_CBM}/image_{i}.png"
    try : 
        result1 = DeepFace.verify(real_image_path_baseline, baseline_image_path ,distance_metric="cosine",enforce_detection=True)
        result_baseline.append(result1["distance"])
        if result1["verified"]:
            same_face_baseline += 1
        total_baseline += 1
    except Exception as e:
        errors_baseline += 1
        # if e contains img2_path print the error
        if "img2_path" in str(e):
            #print(f"Error processing image baseline {i}: {e}")
            result_baseline.append(1)
            fail_baseline += 1
            total_baseline += 1
        #print(f"Error processing image baseline {i}: {e}")
        
    try : 
        result2 = DeepFace.verify(real_image_path_cbm, cbm_image_path ,distance_metric="cosine",enforce_detection=True)
        result_cbm.append(result2["distance"])
        if result2["verified"]:
            same_face_cbm += 1
        total_cbm += 1
    except Exception as e:
        errors_cbm += 1
        if "img2_path" in str(e):
            #print(f"Error processing image CBM {i}: {e}")
            result_cbm.append(1)
            fail_cbm += 1
            total_cbm += 1
        #print(f"Error processing image  CBM{i}: {e}")
        

print("Baseline - Verified:", same_face_baseline/2000,)
print("CBM - Verified:", same_face_cbm / 2000)

print("Baseline - Verified:", same_face_baseline/ total_baseline)
print("CBM - Verified:", same_face_cbm / total_cbm)
print("Baseline - Errors:", errors_baseline)
print("CBM - Errors:", errors_cbm)
print("Baseline :", len(result_baseline))
print("CBM ", len(result_cbm))
print("Baseline - Average Distance/Similarity:", sum(result_baseline)/len(result_baseline))
print("CBM - Average Distance/Similarity:", sum(result_cbm)/len(result_cbm))
print("##########################################################")
print("Baseline - Failed (No Face Detected):", fail_baseline)
print("CBM - Failed (No Face Detected):", fail_cbm)
print("Baseline - Total Processed:", total_baseline)
print("CBM - Total Processed:", total_cbm)
# This function compares two images and returns if they are the same
"""result1 = DeepFace.verify("images/real.png", "images/baseline.png", distance_metric="cosine",enforce_detection=False)
result2 = DeepFace.verify("images/real.png", "images/cbm.png", distance_metric="cosine",enforce_detection=False)

print("Verified:", result1["verified"])
print("Verified:", result2["verified"])

print("Distance/Similarity:", result1["distance"])
print("Distance/Similarity:", result2["distance"])

print("detected", result1)
print("detected", result2)

"""
