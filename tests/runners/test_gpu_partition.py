"""R31: co-located engines partition a node's or container's GPUs, they do not share them.

A vLLM-class engine pre-allocates most of each device it sees (`--gpu-memory-utilization`), so two co-located
engines that both see every GPU fail with out-of-memory. Per phase and per node the requested GPUs are therefore
the **sum** over the engines (x replicas) placed there — the job asks for the maximum of that over the phases,
plus the coordinator's own GPUs — and every co-located engine process gets a disjoint `CUDA_VISIBLE_DEVICES`
slice (:func:`~rcp_ndcg.runners.script.device_slices`), with the coordinator's devices reserved first. An engine
that declares no GPUs gets the empty slice, never all of them.

The two runners partition differently, per the SLURM docs:

- **Kubernetes** — a phase's engines run as processes of one container, so the container's `nvidia.com/gpu` limit
  is the sum and the runner renders one slice per engine inside it.
- **SLURM** — the engine steps run under `srun --overlap`, which per srun(1) "allows steps to share all resources
  (CPUs, memory, and GRES) with all other steps": SLURM's own per-step device grants are not a disjointness
  guarantee there. The runner instead pins each role's replicas to a disjoint slice of the allocation's nodes
  (one replica per node), so no two engine processes ever share a node, sizes each step's `--gres` to its own
  engine, and lets SLURM's per-step `CUDA_VISIBLE_DEVICES` stand (gres.html, "GPU Management": it is set per step
  and "ensures that the resources assigned to each are unique"). A node's GPU request is the sum of what runs on
  it — the coordinator's own GPUs on the first node plus its engine's.
"""

from __future__ import annotations

import yaml

from rcp_ndcg.runners import KubernetesRunner, SlurmRunner
from rcp_ndcg.runners.base import JobPhase, JobSpec
from rcp_ndcg.runners.script import device_slices
from rcp_ndcg.support.serve import ServeConfig


def _resources(gpus: int):
    from rcp_ndcg.runners import Resources

    return Resources(gpus=gpus)


#: The operator's example phase: a 4-GPU judge and a 1-GPU encoder co-located on one node (same image: a phase's
#: engines run in one container).
JUDGE_4 = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command=["vllm", "serve", "org/model", "--host", "0.0.0.0", "--port", "8000"],
    resources=_resources(4),
    port=8000,
)
ENCODER_1 = ServeConfig(
    image="vllm/vllm-openai:v0.30.0",
    command=["python3", "-m", "encoder", "--host", "0.0.0.0", "--port", "8001"],
    resources=_resources(1),
    port=8001,
)


def _phased_job(*phases: JobPhase) -> JobSpec:
    return JobSpec(name="j", phases=phases)


def test_the_partitioning_function_slices_replicas_in_order() -> None:
    """Two replicas of a 2-GPU engine on one node: the second replica continues after the first (0,1 / 2,3)."""
    assert device_slices([2, 2]) == ["0,1", "2,3"]
    assert device_slices([4, 1]) == ["0,1,2,3", "4"]
    assert device_slices([4, 1], reserved=2) == ["2,3,4,5", "6"]  # the coordinator's own devices come first
    assert device_slices([0, 2]) == ["", "0,1"]  # an engine without GPUs sees none, never all of them


def test_kubernetes_partitions_its_container_among_co_located_engines() -> None:
    """judge (4 GPUs) + encoder (1 GPU) in one phase container: a request of 5 and the slices 0,1,2,3 / 4."""
    phases = (JobPhase(engines={"judge": JUDGE_4, "encoder": ENCODER_1}, argv=("a",)),)
    (job_obj,) = list(yaml.safe_load_all(KubernetesRunner().render([_phased_job(*phases)])["j"]))
    (container,) = job_obj["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 5
    cuda = [line for line in container["command"][2].splitlines() if "CUDA_VISIBLE_DEVICES" in line]
    assert cuda == ["export CUDA_VISIBLE_DEVICES=0,1,2,3", "export CUDA_VISIBLE_DEVICES=4"]


def test_the_coordinators_gpus_are_reserved_ahead_of_the_engines() -> None:
    phases = (JobPhase(engines={"judge": JUDGE_4, "encoder": ENCODER_1}, argv=("a",)),)
    job = JobSpec(name="j", resources=_resources(2), phases=phases)
    (job_obj,) = list(yaml.safe_load_all(KubernetesRunner().render([job])["j"]))
    (container,) = job_obj["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 7  # the coordinator's 2 + 4 + 1
    cuda = [line.split("=", 1)[1] for line in container["command"][2].splitlines() if "CUDA_VISIBLE_DEVICES" in line]
    assert cuda == ["2,3,4,5", "6"]


def test_two_engines_of_two_gpus_each_slice_one_container() -> None:
    a = ServeConfig(image="i:1", command=["x", "--port", "8000"], resources=_resources(2), port=8000)
    b = ServeConfig(image="i:1", command=["y", "--port", "8001"], resources=_resources(2), port=8001)
    phases = (JobPhase(engines={"judge": a, "encoder": b}, argv=("a",)),)
    (job_obj,) = list(yaml.safe_load_all(KubernetesRunner().render([_phased_job(*phases)])["j"]))
    (container,) = job_obj["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4
    cuda = [line.split("=", 1)[1] for line in container["command"][2].splitlines() if "CUDA_VISIBLE_DEVICES" in line]
    assert cuda == ["0,1", "2,3"]


def test_kubernetes_partitions_among_replicas_of_one_engine() -> None:
    """Two replicas of one 2-GPU engine, placed on one node by a direct job spec: disjoint slices and ports."""
    from rcp_ndcg.runners import KubernetesRunner

    a = ServeConfig(image="i:1", command=["x", "--port", "8000"], resources=_resources(2), replicas=2, port=8000)
    phases = (JobPhase(engines={"judge": a}, argv=("a",)),)
    runner = KubernetesRunner()
    # Several replicas of one role are run-scoped StatefulSet pods on Kubernetes, one pod per replica, each with
    # the engine's own two devices -- no two engine processes share a device set, so no slice is rendered.
    assert device_slices([_resources(2).gpus] * 2) == ["0,1", "2,3"]  # the slice continuation, for one node's set
    (job_obj, stateful_set, _service) = list(yaml.safe_load_all(runner.render([_phased_job(*phases)])["j"]))
    assert stateful_set["spec"]["replicas"] == 2
    (engine,) = stateful_set["spec"]["template"]["spec"]["containers"]
    assert engine["resources"]["limits"] == {"nvidia.com/gpu": 2}  # each pod: its own two devices
    (container,) = job_obj["spec"]["template"]["spec"]["containers"]
    assert "CUDA_VISIBLE_DEVICES" not in container["command"][2]  # the phase waits for the pods; it runs none


def test_a_gpu_less_engine_sees_no_device() -> None:
    none = ServeConfig(
        image="vllm/vllm-openai:v0.30.0", command=["x", "--port", "8001"], resources=_resources(0), port=8001
    )
    phases = (JobPhase(engines={"judge": JUDGE_4, "encoder": none}, argv=("a",)),)
    (job_obj,) = list(yaml.safe_load_all(KubernetesRunner().render([_phased_job(*phases)])["j"]))
    (container,) = job_obj["spec"]["template"]["spec"]["containers"]
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 4  # 4 + 0: a GPU-less engine adds nothing
    cuda = [
        line.split("=", 1)[1].strip("'")
        for line in container["command"][2].splitlines()
        if "CUDA_VISIBLE_DEVICES" in line
    ]
    assert cuda == ["0,1,2,3", ""]  # the empty slice, never all of them


def test_the_coordinators_gpus_are_reserved_on_slurm_too() -> None:
    """The coordinator's task asks its own --gres: SLURM's per-step CUDA_VISIBLE_DEVICES is unique per step
    (gres.html), so the disjoint remainder of the coordinator's node is its reservation -- the co-located
    engine's step, launched with its own --gres, cannot take them."""
    phases = (JobPhase(engines={"judge": JUDGE_4, "encoder": ENCODER_1}, argv=("a",)),)
    job = JobSpec(name="j", resources=_resources(2), phases=phases)
    script = SlurmRunner(container_runtime="pyxis").render([job])["j"]
    assert "#SBATCH --gres=gpu:6" in script  # the maximum over the nodes
    assert "srun --overlap --nodes=1 --ntasks=1 --nodelist=${RCP_NDCG_HOSTS[0]} --gres=gpu:2 " in script
    # No CUDA export for the coordinator: the step's own grant names the disjoint remainder, and a hardcoded
    # prefix would override it with a co-located engine's devices.


def test_a_gpu_less_slurm_engine_sees_no_device() -> None:
    """An engine without GPUs gets the empty CUDA_VISIBLE_DEVICES: its step carries no --gres, and srun(1)
    allocates such a step all of the job's GRES -- never all of it, per the partitioning's contract."""
    none = ServeConfig(image=None, command=["x", "--port", "8001"], resources=_resources(0), port=8001)
    judge = ServeConfig(
        image=None, command=["vllm", "serve", "m", "--port", "8000"], resources=_resources(4), port=8000
    )
    phases = (JobPhase(engines={"judge": judge, "encoder": none}, argv=("a",)),)
    script = SlurmRunner().render([_phased_job(*phases)])["j"]
    cuda = [line.split("=", 1)[1] for line in script.splitlines() if line.startswith("export CUDA_VISIBLE_DEVICES=")]
    assert cuda == ["''"]  # the empty slice for the GPU-less engine; the judge keeps SLURM's per-step grant
    assert (
        "--gres=gpu:4 " in script and "CUDA_VISIBLE_DEVICES" not in script.split("ENGINE_JUDGE")[0].split("ENGINE_")[-1]
    )


def test_an_engine_free_phase_still_carries_the_coordinators_gpus() -> None:
    """The coordinator runs in every phase: an engine-free phase's node requests its GPUs alone."""
    from rcp_ndcg.runners import Resources

    phases = (JobPhase(argv=("a",)),)
    job = JobSpec(name="j", resources=Resources(gpus=2), phases=phases)
    script = SlurmRunner().render([job])["j"]
    assert "#SBATCH --ntasks=1\n" in script and "#SBATCH --gres=gpu:2\n" in script


def test_slurm_asks_for_the_sum_of_what_a_node_hosts_and_the_maximum_over_phases() -> None:
    """The judge (4 GPUs) shares the coordinator's node; the encoder runs alone on the next; a 6-GPU engine in
    another phase sets the job's maximum --gres."""
    judge = JUDGE_4
    encoder = ENCODER_1
    big = ServeConfig(
        image="vllm/vllm-openai:v0.30.0", command=["y", "--port", "8001"], resources=_resources(6), port=8001
    )
    phases = (
        JobPhase(engines={"judge": judge, "encoder": encoder}, argv=("a",)),
        JobPhase(engines={"encoder": big}, argv=("b",)),
    )
    script = SlurmRunner(container_runtime="pyxis").render([_phased_job(*phases)])["j"]
    assert "#SBATCH --nodes=2\n" in script  # the largest phase's replica total
    assert "#SBATCH --gres=gpu:6\n" in script  # the maximum over the phases' per-node sums (4, then 6)
    assert "--gres=gpu:4 " in script and "--gres=gpu:1 " in script and "--gres=gpu:6 " in script
    assert "CUDA_VISIBLE_DEVICES" not in script  # SLURM's per-step value stands: no two engines share a node
    # The largest engine shares the coordinator's node; the coordinator pins itself to it.
    assert 'HOSTS_JUDGE=("${RCP_NDCG_HOSTS[@]:0:1}")\n' in script
    assert 'HOSTS_ENCODER=("${RCP_NDCG_HOSTS[@]:1:1}")\n' in script
    assert "srun --overlap --nodes=1 --ntasks=1 --nodelist=${RCP_NDCG_HOSTS[0]} " in script
