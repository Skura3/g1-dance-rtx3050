"""Measure actual mjlab G1 stepping throughput, excluding compilation and rendering."""
import argparse,json,time,subprocess,threading
from pathlib import Path
from common import ROOT,add_sources

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--envs',type=int,default=16);p.add_argument('--steps',type=int,default=160);p.add_argument('--threads',type=int,default=2);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    add_sources();import numpy as np,torch,mujoco,warp as wp
    from mjlab.asset_zoo.robots import get_g1_robot_cfg
    from mjlab.scene import Scene,SceneCfg
    from mjlab.sim import Simulation,SimulationCfg,MujocoCfg
    from mjlab.terrains import TerrainEntityCfg
    torch.set_num_threads(a.threads);torch.set_num_interop_threads(1);wp.init();samples=[];stop=threading.Event()
    def monitor():
        while not stop.is_set():
            raw=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,temperature.gpu,utilization.gpu','--format=csv,noheader,nounits'],text=True)
            samples.append([float(x.strip()) for x in raw.split(',')]);stop.wait(.25)
    thread=threading.Thread(target=monitor,daemon=True);thread.start();build=time.perf_counter()
    try:
        scene=Scene(SceneCfg(num_envs=a.envs,terrain=TerrainEntityCfg(terrain_type='plane'),entities={'robot':get_g1_robot_cfg()},extent=2.),device='cuda:0')
        sim=Simulation(num_envs=a.envs,cfg=SimulationCfg(nconmax=64,njmax=256,graph_conditional=False,mujoco=MujocoCfg(timestep=.005,iterations=10,ls_iterations=20)),spec=scene.spec,device='cuda:0')
        scene.initialize(mj_model=sim.mj_model,model=sim.model,data=sim.data);sim.reset();scene.reset();robot=scene['robot'];m=sim.mj_model
        host=mujoco.MjData(m);mujoco.mj_resetDataKeyframe(m,host,m.key('init_state').id)
        equilibrium=np.load(ROOT/'outputs/environment_validation/standing_controller.npz')
        q=torch.tensor(equilibrium['q0'],dtype=torch.float32,device='cuda:0');u=torch.tensor(equilibrium['u0'],dtype=torch.float32,device='cuda:0')
        sim.data.qpos[:]=q;sim.data.ctrl[:]=u;sim.forward()
        for _ in range(30):sim.step()
        wp.synchronize();build_s=time.perf_counter()-build;durations=[]
        for repeat in range(3):
            sim.reset();sim.data.qpos[:]=q;sim.data.qvel.zero_();sim.data.ctrl[:]=u;sim.forward();wp.synchronize();t=time.perf_counter()
            for _ in range(a.steps):sim.step()
            wp.synchronize();durations.append(time.perf_counter()-t)
        finite=bool(torch.isfinite(sim.data.qpos).all() and torch.isfinite(sim.data.qvel).all())
        median=float(np.median(durations));report={'num_envs':a.envs,'cpu_threads':a.threads,'steps_per_repeat':a.steps,'repeat_seconds':durations,'warmup_build_s':build_s,'median_world_steps_per_s':a.steps/median,'aggregate_env_steps_per_s':a.envs*a.steps/median,'aggregate_sim_seconds_per_wall_second':a.envs*a.steps*.005/median,'cuda_graphs_enabled':sim.use_cuda_graph,'finite':finite,'torch_peak_allocated_mib':torch.cuda.max_memory_allocated()/2**20,'torch_peak_reserved_mib':torch.cuda.max_memory_reserved()/2**20,'gpu_free_mib':torch.cuda.mem_get_info()[0]/2**20,'scope':'Simulation stepping only; fixed PD targets can fall. This does not benchmark training or policy convergence.'}
    finally:stop.set();thread.join(timeout=2.)
    arr=np.asarray(samples);report.update(gpu_total_memory_used_peak_mib=float(arr[:,0].max()),gpu_temperature_max_c=float(arr[:,1].max()),gpu_utilization_peak_percent=float(arr[:,2].max()))
    a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))

if __name__=='__main__':main()
