A reimplementation of spm_realign.m in python.

It is almost entirely the same as the original available on [spm12's github page]( https://github.com/spm/spm12) (except for some functionalities I didn't need, masking, register to mean...).

## use

```python
from realign import realign, spm_imatrix, plot_params
import glob

files = sorted(glob.glob("*.nii"))
P = realign(files)
# P is an array of nibabel Nifti1Image with corrected affines

# decompose matrices:
Params = np.zeros((len(P),12))
for i in range(len(P)):
    Params[i,:] = spm_imatrix(P[i].affine @ np.linalg.pinv(P[0].affine))
translations = Params[:, :3]                # shape (nvols, 3) -> x, y, z
rotations    = Params[:, 3:6] * 180 / np.pi # shape (nvols, 3) -> pitch, roll, yaw

# or just plot parameters:
plot_params(P)
```

## comparison

| spm (matlab)                | this repo (python)          |
|-----------------------------|-----------------------------|
| ![matlab](/comp/matlab.png) | ![python](/comp/python.png) |
