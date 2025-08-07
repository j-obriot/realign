import nibabel as nib
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.interpolate import NdBSpline
import matplotlib.pyplot as plt

def chain_and(*args):
    A = np.ones(args[0].shape, dtype=bool)
    for i in args:
        A = np.logical_and(A, i)
    return A

def corr_affine(A):
    A = np.nan_to_num(A)
    A[3,3] = 1
    return A

def afn(nii, fn, *args, **kwargs):
    a = fn(nii.get_fdata(), *args, **kwargs)
    return nib.Nifti1Image(a, nii.affine)

def spm_imatrix(M):
    R         = M[:3,:3]
    C         = np.linalg.cholesky(R.T @ R)
    P         = [*M[:3,3], 0, 0, 0, *np.diag(C), 0, 0, 0]
    if np.linalg.det(R) < 0:
        P[6] = -P[6]

    C         = np.linalg.solve(np.diag(np.diag(C)), C)
    P[9:12]  = C.flatten()[[3, 6, 7]]
    R0        = spm_matrix([0, 0, 0, 0, 0, 0, *P[6:12]])
    R0        = R0[:3,:3]
    # R1        = np.linalg.solve(R, R0)
    R1        = R @ np.linalg.pinv(R0)

    rang      = lambda x: np.min(np.max(x, -1), 0)

    P[4]      = np.asin(rang(R1[0,2]))
    if (abs(P[4])-np.pi/2)**2 < 1e-9:
        P[3]  = 0
        P[5]  = np.atan2(-rang(R1[1,0]), rang(np.linalg.solve(-R1[2,0], R1[0,2])))
    else:
        c     = np.cos(P[4])
        P[3]  = np.atan2(rang(R1[1,2]/c), rang(R1[2,2]/c))
        P[5]  = np.atan2(rang(R1[0,1]/c), rang(R1[0,0]/c))
    return P

def spm_matrix(P):
    if len(P) == 3:
        A = np.eye(4)
        A[:3,3] = P
        return
        
    q  = [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0]
    P  = [*P, *q[len(P):12]]
    
    T = np.eye(4)
    T[:3,3] = P[:3]

    R1  =  np.array([[1,  0           , 0           , 0],
                     [0,  np.cos(P[3]), np.sin(P[3]), 0],
                     [0, -np.sin(P[3]), np.cos(P[3]), 0],
                     [0,  0           , 0           , 1]])

    R2  =  np.array([[ np.cos(P[4]), 0, np.sin(P[4]), 0],
                     [ 0           , 1, 0           , 0],
                     [-np.sin(P[4]), 0, np.cos(P[4]), 0],
                     [ 0           , 0, 0           , 1]])

    R3  =  np.array([[ np.cos(P[5]), np.sin(P[5]), 0, 0],
                     [-np.sin(P[5]), np.cos(P[5]), 0, 0],
                     [ 0           , 0           , 1, 0],
                     [ 0           , 0           , 0, 1]])

    R   = R1 @ R2 @ R3

    Z = np.diag([*P[6:9], 1])

    S   =  np.array([[1, P[9], P[10], 0],
                     [0, 1   , P[11], 0],
                     [0, 0   , 1    , 0],
                     [0, 0   , 0    , 1]])
    
    A = T @ R @ Z @ S
    return A

def coords(p, M1, M2, x1, x2, x3):
    M  = np.linalg.inv(M2) @ np.linalg.inv(spm_matrix(p)) @ M1
    y1 = M[0,0]*x1 + M[0,1]*x2 + M[0,2]*x3 + M[0,3]
    y2 = M[1,0]*x1 + M[1,1]*x2 + M[1,2]*x3 + M[1,3]
    y3 = M[2,0]*x1 + M[2,1]*x2 + M[2,2]*x3 + M[2,3]
    return y1, y2, y3
    
def make_A(M, x1, x2, x3, dG1, dG2, dG3, wt, lkp):
    p0 = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)
    A  = np.zeros((len(x1), len(lkp)))
    for i in range(len(lkp)):
        pt         = p0.copy()
        pt[lkp[i]] = pt[i] + 1e-6
        y1, y2, y3 = coords(pt, M, M, x1, x2, x3)
        tmp        = np.sum(np.array([y1-x1, y2-x2, y3-x3]) * np.array([dG1, dG2, dG3]), axis=0) / (-1e-6)
        A[:,i] = tmp
    return A


def realign(P,
            quality=0.9,
            interp=2,
            wrap=(0,0,0), # ununsed
            sep=4,
            fwhm=5,
            rtm=1 # useless
            ):
    lkp = np.arange(6)

    for i in range(len(P)):
        P[i] = nib.load(P[i])


    skip = 1 / np.linalg.norm(P[0].affine[:3,:3], axis=0) * sep
    d    = P[0].shape[:3]
    np.random.seed(0)

    # x1,x2,x3 = np.mgrid[0:d[0]-.5:skip[0], 0:d[1]-.5:skip[1], 0:d[2]-.5:skip[2]].transpose((0, 3, 2, 1))
    x1,x2,x3 = np.meshgrid(np.arange(0, d[0], skip[0]),
                           np.arange(0, d[1], skip[1]),
                           np.arange(0, d[2], skip[2]),
                           indexing='ij')
    x1 = x1 + np.random.rand(*x1.shape)*0.5
    x2 = x2 + np.random.rand(*x2.shape)*0.5
    x3 = x3 + np.random.rand(*x3.shape)*0.5

    x1 = x1.flatten()
    x2 = x2.flatten()
    x3 = x3.flatten()

    # no masking, possibly TODO
    wt = []

    # V   = smooth_vol(P(1),flags.interp,flags.wrap,flags.fwhm);

    vx = np.linalg.norm(P[0].affine[:3,:3], axis=0)
    s  = (1 / vx) * (fwhm / np.sqrt(8*np.log(2)))

    V = afn(P[0], gaussian_filter, s, axes=(0, 1, 2))

    # not completely sure about this
    # [G,dG1,dG2,dG3] = spm_bsplins(V,x1,x2,x3,deg);
    db = NdBSpline(tuple(np.arange(V.shape[i] + interp+1)-interp//2 for i in range(3)), V.get_fdata(), [interp] * 3)
    G = db(np.array([x1, x2, x3]).transpose((1, 0)))
    [dG1, dG2, dG3] = [db(np.array([x1, x2, x3]).transpose((1, 0)), nu=nu) for nu in np.eye(3, dtype=int)]

    A0  = make_A(P[0].affine, x1, x2, x3, dG1, dG2, dG3, wt, lkp)

    b   = G.copy()

    dt = []
    # Remove voxels that contribute very little to the final estimate
    if len(P) > 2:
        Alpha = np.hstack([A0, b[:, None]])
        Alpha = Alpha.T @ Alpha
        det0  = np.linalg.det(Alpha)
        det1  = det0
        while det1 / det0 > quality:
            dt.append(det1 / det0)
            dets = np.zeros((A0.shape[0],))
            for i in range(A0.shape[0]):
                tmp     = np.hstack([A0[i,:], b[i]])
                dets[i] = np.linalg.det(Alpha - tmp[:, None].T @ tmp)
            msk = np.argsort(det1 - dets)
            msk        = msk[:round(len(dets)/10)]

            A0 = np.delete(A0, msk, axis=0)
            b = np.delete(b, msk)
            G = np.delete(G, msk)
            x1 = np.delete(x1, msk)
            x2 = np.delete(x2, msk)
            x3 = np.delete(x3, msk)
            dG1 = np.delete(dG1, msk)
            dG2 = np.delete(dG2, msk)
            dG3 = np.delete(dG3, msk)

            Alpha = np.hstack([A0, b[:, None]])
            Alpha = Alpha.T @ Alpha
            det1  = np.linalg.det(Alpha)


    if rtm:
        count = np.ones(b.shape)
        ave   = G.copy()
        grad1 = dG1.copy()
        grad2 = dG2.copy()
        grad3 = dG3.copy()


    for i in range(1, len(P)):
        V = afn(P[i], gaussian_filter, s, axes=(0, 1, 2))
        d  = V.shape[:3]
        ss = np.inf
        countdown = -1
        for iteration in range(64):
            y1, y2, y3 = coords([0, 0, 0, 0, 0, 0], P[0].affine, P[i].affine, x1, x2, x3)
            msk = chain_and(y1 >= 0, y1 <= d[0], y2 >= 0, y2 <= d[1], y3 >= 0, y3 <= d[2])

            if len(msk)<32:
                print("ERROR")
                # return

            # F          = spm_bsplins(V, y1(msk),y2(msk),y3(msk),deg);
            db = NdBSpline(tuple(np.arange(V.shape[i] + interp+1)-interp//2 for i in range(3)), V.get_fdata(), [interp] * 3)
            F = db(np.array([y1[msk], y2[msk], y3[msk]]).transpose((1, 0)))

            A          = A0[msk, :]
            b1         = b[msk]
            sc         = b1.sum() / F.sum()
            b1         = b1 - F * sc
            # soln       = (A'*A)\(A'*b1)
            soln = np.linalg.solve(A.T @ A, A.T @ b1)

            p           = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)
            p[lkp]      = p[lkp] + soln.T
            # P[i].affine = np.linalg.solve(spm_matrix(p), P[i].affine)
            P[i] = nib.Nifti1Image(P[i].get_fdata(),
                                   corr_affine(np.linalg.solve(spm_matrix(p), P[i].affine)))

            pss        = ss
            ss         = np.sum(b1**2) / len(b1)
            if (pss - ss) / pss < 1e-8 and countdown == -1:
                countdown = 2
            if countdown != -1:
                if countdown == 0:
                    break
                countdown -= 1

        if rtm:
            tiny = 5e-2
            msk = chain_and(y1 >= 1 - tiny, y1 <= d[0] + tiny,
                            y2 >= 1 - tiny, y2 <= d[1] + tiny,
                            y3 >= 1 - tiny, y3 <= d[2] + tiny)

            count[msk] = count[msk] + 1
            db = NdBSpline(tuple(np.arange(V.shape[i] + interp+1)-interp//2 for i in range(3)), V.get_fdata(), [interp] * 3)
            G = db(np.array([y1[msk], y2[msk], y3[msk]]).transpose((1, 0)))
            [dG1, dG2, dG3] = [db(np.array([y1[msk], y2[msk], y3[msk]]).transpose((1, 0)), nu=nu) for nu in np.eye(3, dtype=int)]
            ave[msk]   = ave[msk]   +   G*sc
            grad1[msk] = grad1[msk] + dG1*sc
            grad2[msk] = grad2[msk] + dG2*sc
            grad3[msk] = grad3[msk] + dG3*sc

    return P

    # # Register to mean
    # M  = P[0].affine
    # A0 = make_A(M,x1,x2,x3,grad1/count,grad2/count,grad3/count,wt,lkp)
    # 
    # b = ave / count
    # 
    # for i in range(len(P)):
    #     V = afn(P[i], gaussian_filter, s, axes=(0, 1, 2))
    #     d  = V.shape[:3]
    #     ss = np.inf
    #     countdown = -1
    #     for iteration in range(64):
    #         y1, y2, y3 = coords([0, 0, 0, 0, 0, 0], M, P[i].affine, x1, x2, x3)
    #         msk = chain_and(y1 >= 0, y1 <= d[0], y2 >= 0, y2 <= d[1], y3 >= 0, y3 <= d[2])
    # 
    #         if len(msk)<32:
    #             print("ERROR")
    #             # return
    # 
    #         db = NdBSpline(tuple(np.arange(V.shape[i] + interp+1)-interp//2 for i in range(3)), V.get_fdata(), [interp] * 3)
    #         F = db(np.array([y1[msk], y2[msk], y3[msk]]).transpose((1, 0)))
    # 
    #         A          = A0[msk, :]
    #         b1         = b[msk]
    #         sc         = b1.sum() / F.sum()
    #         b1         = b1 - F * sc
    #         soln = np.linalg.solve(A.T @ A, A.T @ b1)
    # 
    #         p           = np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0], dtype=np.float64)
    #         p[lkp]      = p[lkp] + soln.T
    #         # P[i].affine = np.linalg.solve(spm_matrix(p), P[i].affine)
    #         P[i] = nib.Nifti1Image(P[i].get_fdata(),
    #                                corr_affine(np.linalg.solve(spm_matrix(p), P[i].affine)))
    # 
    #         pss        = ss
    #         ss         = np.sum(b1**2) / len(b1)
    #         if (pss - ss) / pss < 1e-8 and countdown == -1:
    #             countdown = 2
    #         if countdown != -1:
    #             if countdown == 0:
    #                 break
    #             countdown -= 1
    # 
    # 
    # 
    # # Aligning everything to the first image
    # M = M / P[0].affine
    # for i in range(len(P)):
    #     P[i] = nib.Nifti1Image(P[i].get_fdata(),
    #                            corr_affine(M @ P[i].affine))



# ----------- TEST -----------

def plot_params(P):
    Params = np.zeros((len(P),12))
    for i in range(len(P)):
        Params[i,:] = spm_imatrix(P[i].affine @ np.linalg.pinv(P[0].affine))
    Params = Params - Params[len(P)//2, :]

    fig, axes = plt.subplots(2, 1)
    axes = axes.flatten()
    axes[0].plot(Params[:,:3], label=['x translation','y translation','z translation'])
    axes[0].legend()
    axes[0].set_ylabel('mm')
    axes[0].set_xlabel('image')
    axes[0].set_ylim([-0.55, 0.55])
    axes[0].set_xlim([0, len(P)])
    axes[0].set_yticks([-0.5, 0, 0.5])
    axes[0].grid()


    axes[1].plot(Params[:,3:6]*180/np.pi, label=['pitch','roll','yaw'])
    axes[1].legend()
    axes[1].set_ylabel('degrees')
    axes[1].set_xlabel('image')
    axes[1].set_ylim([-1.2, 1.2])
    axes[1].set_xlim([0, len(P)])
    axes[1].set_yticks(np.arange(-1, 1.1, 0.5))
    axes[1].grid()

    plt.show()
