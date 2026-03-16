from deepface import DeepFace



metrics = ["cosine", "euclidean", "euclidean_l2", "angular"]
# This function compares two images and returns if they are the same
result1 = DeepFace.verify("images/real.png", "images/baseline.png", distance_metric="cosine",enforce_detection=False)
result2 = DeepFace.verify("images/real.png", "images/cbm.png", distance_metric="cosine",enforce_detection=False)

print("Verified:", result1["verified"])
print("Verified:", result2["verified"])

print("Distance/Similarity:", result1["distance"])
print("Distance/Similarity:", result2["distance"])

print("detected", result1)
print("detected", result2)


