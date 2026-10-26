"""
Direction check for the retrained Air75 policy in the L2F simulator.

Simulates an Air75-like quadrotor (one of the 300 training drones) and moves the position
target +x (forward), +y (left), +z (up) and back, so each axis can be checked visually.
Conventions are FLU: x = forward, y = left, z = up.

Needs the Python packages of the RAPTOR repo (l2f, foundation_policy, ui-server; see its README).
Usage:
    terminal 1:  ui-server                     (then open http://localhost:13337)
    terminal 2:  python air75_demo.py
    without UI:  python air75_demo.py --headless
    the delay/dropout post-trained policy:  python air75_demo.py --policy ../policy/air75_delay_dropout/checkpoint.h5
"""
import argparse, asyncio, json, pathlib
from copy import copy
import numpy as np
import l2f
from l2f import vector1 as vector
import foundation_policy

POLICY_DIR = pathlib.Path(__file__).resolve().parent.parent / "policy"
DEFAULT_POLICY = POLICY_DIR / "air75_base/checkpoint.h5"
# 6.json: 41.9 g, 38.8 mm arm, thrust-to-weight 4.9 (closest of the 300 training drones to the Air75 II)
DEFAULT_DRONE = POLICY_DIR / "dynamics_parameters/6.json"

STEP = 0.3  # m
SEGMENT_SECONDS = 3
TARGETS = [
    ("hover at origin", [0, 0, 0]),
    ("+x  (forward)", [STEP, 0, 0]),
    ("back to origin", [0, 0, 0]),
    ("+y  (left)", [0, STEP, 0]),
    ("back to origin", [0, 0, 0]),
    ("+z  (up)", [0, 0, STEP]),
    ("back to origin", [0, 0, 0]),
]

parser = argparse.ArgumentParser()
parser.add_argument("--policy", default=str(DEFAULT_POLICY), help="checkpoint.h5 to fly ('original' for the published RAPTOR policy)")
parser.add_argument("--drone", default=str(DEFAULT_DRONE), help="dynamics parameters JSON of the simulated drone")
parser.add_argument("--headless", action="store_true", help="no UI, just print the result per segment")
args = parser.parse_args()

policy = foundation_policy.Raptor() if args.policy == "original" else foundation_policy.load(args.policy)
device = l2f.Device()
rng = vector.VectorRng()
env = vector.VectorEnvironment()
ui = l2f.UI()
params = vector.VectorParameters()
state = vector.VectorState()
next_state = vector.VectorState()
observation = np.zeros((env.N_ENVIRONMENTS, env.OBSERVATION_DIM), dtype=np.float32)

vector.initialize_rng(device, rng, 0)
vector.initialize_environment(device, env)
vector.sample_initial_parameters(device, env, params, rng)
l2f.parameters_from_json(device, env.environments[0], open(args.drone).read(), params.parameters[0])
vector.initial_state(device, env, params, state)
dynamics = json.loads(l2f.parameters_to_json(device, env.environments[0], params.parameters[0]))["dynamics"]
print(f"policy: {args.policy}")
print(f"drone:  {args.drone}  (mass {dynamics['mass'] * 1000:.1f} g)")


async def run(websocket=None):
    policy.reset()
    dt = 0.01
    for name, target in TARGETS:
        target = np.array(target, dtype=np.float32)
        for _ in range(int(SEGMENT_SECONDS / dt)):
            vector.observe(device, env, params, state, observation, rng)
            obs = observation[:, :22].copy()
            obs[:, :3] -= target  # the policy flies to the origin of its position input, so shift it to move the target
            action = policy.evaluate_step(obs)
            dts = vector.step(device, env, params, state, action, next_state, rng)
            state.assign(next_state)
            dt = dts[-1]
            if websocket is not None:
                await websocket.send(vector.set_state_action_message(device, env, params, ui, copy(state), action))
                await asyncio.sleep(dt)
        position = np.array(state.states[0].position)
        error = np.linalg.norm(position - target)
        print(f"{name:16s} target {np.round(target, 2)}  reached {np.round(position, 3)}  error {error * 100:5.1f} cm  {'OK' if error < 0.05 else 'FAIL'}")


async def main():
    if args.headless:
        return await run()
    import websockets
    async with websockets.connect("ws://localhost:13337/backend") as websocket:
        handshake = json.loads(await websocket.recv())
        assert handshake["channel"] == "handshake"
        ui.ns = handshake["data"]["namespace"]
        await websocket.send(vector.set_ui_message(device, env, ui))
        await websocket.send(vector.set_parameters_message(device, env, params, ui))
        await asyncio.sleep(1)
        await run(websocket)


if __name__ == "__main__":
    asyncio.run(main())
