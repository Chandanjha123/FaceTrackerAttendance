import pickle
import numpy as np

with open('data/faces_data.pkl', 'rb') as f:
    faces = pickle.load(f)

with open('data/names.pkl', 'rb') as f:
    names = pickle.load(f)

print("✅ Face Data Shape:", np.array(faces).shape)
print("✅ Number of Labels:", len(names))
print("🧍‍♂️ First Label:", names[0] if names else "None")
