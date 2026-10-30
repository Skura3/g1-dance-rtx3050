"""Inverse dynamics plus finite-horizon LQR for simulator standing validation.
Original actuator torque limits, free base and gravity remain active.
"""
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def build_controller(model, initial):
    data=mujoco.MjData(model)
    addresses=model.jnt_qposadr[model.actuator_trnid[:,0]]
    def residual(x):
        mujoco.mj_resetData(model,data);data.qpos[:]=initial;data.qpos[2]=x[0]
        data.qpos[3:7]=Rotation.from_euler('xy',x[1:3]).as_quat()[[3,0,1,2]]
        data.qacc[:]=0.;mujoco.mj_inverse(model,data)
        return data.qfrc_inverse[:6].copy()
    fit=least_squares(residual,[initial[2]-.004,0.,.006],
        bounds=([initial[2]-.03,-.05,-.05],[initial[2],.05,.05]),
        max_nfev=1000,xtol=1e-13,ftol=1e-13,gtol=1e-13)
    residual(fit.x);q0=data.qpos.copy()
    u0=q0[addresses]+data.qfrc_inverse[model.jnt_dofadr[model.actuator_trnid[:,0]]]/model.actuator_gainprm[:,0]
    data.ctrl[:]=u0;mujoco.mj_forward(model,data)
    error=float(np.max(np.abs(data.qacc)))
    print(f'Standing equilibrium max acceleration: {error:.6g}',flush=True)
    if error>1e-3:raise RuntimeError('Could not find static G1 equilibrium')
    A=np.zeros((2*model.nv,2*model.nv));B=np.zeros((2*model.nv,model.nu))
    mujoco.mjd_transitionFD(model,data,1e-6,True,A,B,None,None)
    Q=np.diag(np.concatenate([[200.,200.,200.,500.,500.,100.],
        np.full(model.nv-6,10.),np.full(model.nv,1.)]))
    R=np.eye(model.nu)*.5;P=Q.copy()
    for _ in range(500):
        K=np.linalg.solve(R+B.T@P@B,B.T@P@A)
        P=Q+A.T@P@(A-B@K);P=(P+P.T)/2
    info={'equilibrium_max_acceleration':error,'equilibrium_solver_evaluations':fit.nfev,
          'feedback_horizon_steps':500,
          'closed_loop_spectral_radius':float(np.max(np.abs(np.linalg.eigvals(A-B@K)))),
          'method':'Inverse-dynamics contact equilibrium + finite-horizon LQR through original PD actuators'}
    return q0,u0,K,info


def control(model,qpos,qvel,q0,u0,K):
    dq=np.zeros(model.nv)
    mujoco.mj_differentiatePos(model,dq,1.,q0,np.asarray(qpos,dtype=float))
    return u0-K@np.concatenate([dq,qvel])
