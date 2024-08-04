import numpy as np

lead_I = np.array([...]) 
lead_II = np.array([...])

aVR = -(lead_I + lead_II) / 2
aVL = lead_I - lead_II / 2
aVF = lead_II - lead_I / 2

print("aVR:", aVR)
print("aVL:", aVL)
print("aVF:", aVF)
