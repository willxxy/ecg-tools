import numpy as np

lead_I = np.array([...]) 
lead_II = np.array([...])
lead_III = np.array([...])

# more theoretically correct
aVR = -((I + II) / 2)
aVL = (I - III) / 2
aVF = (II + III) / 2

print("aVR:", aVR)
print("aVL:", aVL)
print("aVF:", aVF)
