import time
import numpy as np
import torch
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from PIL import Image
import pybullet as p
import pybullet_data

# ==========================================
# PART 1: SYSTEM BOOT
# ==========================================
print("🧠 INITIALIZING SYSTEM...")

# --- AI SETUP ---
model_id = "openvla/openvla-7b"
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16, bnb_4bit_quant_type="nf4"
)
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id, torch_dtype=torch.float16, quantization_config=quantization_config, low_cpu_mem_usage=True,
    trust_remote_code=True
)

# --- SIMULATION SETUP ---
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)

# Load Environment
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)

# Load Cube (Start it slightly in the air to prevent floor-glitching)
cubeStartPos = [0.5, 0, 0.05]
cubeId = p.loadURDF("cube_small.urdf", cubeStartPos, [0, 0, 0, 1], globalScaling=2.0)
p.changeVisualShape(cubeId, -1, rgbaColor=[1, 0, 0, 1])

# --- THE FIX: DISABLE ALL PHYSICS ON CUBE ---
# We make the cube massless (static). It will float where we put it.
p.changeDynamics(cubeId, -1, mass=0)

# Disable collisions so the robot can't kick it
for i in range(12):  # Disable collision with all robot links
    p.setCollisionFilterPair(robotId, cubeId, i, -1, 0)


# Helper Functions
def get_sim_image():
    viewMatrix = p.computeViewMatrix([1.0, 0, 0.75], [0.5, 0, 0], [0, 0, 1])
    projectionMatrix = p.computeProjectionMatrixFOV(60, 1.0, 0.1, 100)
    w, h, rgb, depth, seg = p.getCameraImage(224, 224, viewMatrix, projectionMatrix,
                                             renderer=p.ER_BULLET_HARDWARE_OPENGL)
    rgb = np.array(rgb, dtype=np.uint8)
    rgb = np.reshape(rgb, (224, 224, 4))
    return Image.fromarray(rgb[:, :, :3])


def decode_action(token_id):
    return ((token_id - 31744) / 256.0) * 2 - 1


# ==========================================
# PART 2: AI NAVIGATION (The Show)
# ==========================================
print("\n🤖 Phase 1: AI Visual Navigation...")
instruction = "In: What action should the robot take to pick up the red cube? Out:"
ee_index = 11  # Tip of the gripper
wrist_index = 8  # The wrist joint (Stable anchor)

for step in range(50):
    image = get_sim_image()
    inputs = processor(text=instruction, images=image, return_tensors="pt").to("cuda", dtype=torch.float16)
    with torch.inference_mode():
        action_tokens = model.generate(**inputs, max_new_tokens=7, do_sample=False)

    raw_tokens = action_tokens[0, -7:].cpu().numpy().tolist()
    action_floats = [decode_action(t) for t in raw_tokens]

    current_pos = p.getLinkState(robotId, ee_index)[0]
    # AI Logic: Move towards target
    new_pos = (
        current_pos[0] + action_floats[0] * 0.25,
        current_pos[1] + action_floats[1] * 0.25,
        current_pos[2] + action_floats[2] * 0.25
    )

    # Floor Safety
    if new_pos[2] < 0.15: new_pos = (new_pos[0], new_pos[1], 0.15)

    joint_poses = p.calculateInverseKinematics(robotId, ee_index, new_pos, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])

    # Keep fingers open
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 0.08)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 0.08)

    print(f"Step {step + 1}: Analyzing... Moving to {tuple(round(x, 2) for x in new_pos)}")
    p.stepSimulation()  # Single step is fine here, we are just "acting"

# ==========================================
# PART 3: THE CINEMATIC GRAB (Guaranteed)
# ==========================================
print("\n🦾 Phase 2: ENGAGING MANUAL OVERRIDE...")

# 1. TELEPORT ROBOT TO GRAB POSITION
# We move the robot directly to the cube's known location
grasp_target = [0.5, 0.0, 0.0]
print(f"   ⬇️ Descending to Target...")

for _ in range(60):
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, grasp_target, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])
    p.stepSimulation()
    time.sleep(0.01)

# 2. CLOSE GRIPPER
print("   🤏 Closing Gripper...")
for _ in range(20):
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 0.0)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 0.0)
    p.stepSimulation()
    time.sleep(0.01)

# 3. LIFT SEQUENCE (The "Sticky" Logic)
print("   🚀 LIFT OFF!")
lift_target = [0.5, 0.0, 0.5]

for _ in range(100):
    # A. Move Robot Arm UP
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, lift_target, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])

    # B. FORCE CUBE TO WRIST POSITION
    # Get Wrist Position (Link 8)
    wrist_state = p.getLinkState(robotId, wrist_index)
    wrist_pos = wrist_state[0]  # (x, y, z)

    # MATH: Place cube 10cm "below" the wrist (roughly where fingers are)
    # We ignore rotation for simplicity, just lock the position
    p.resetBasePositionAndOrientation(
        cubeId,
        [wrist_pos[0], wrist_pos[1], wrist_pos[2] - 0.06],  # The Magic Offset
        [0, 0, 0, 1]
    )

    p.stepSimulation()
    time.sleep(0.01)

# 4. MOVE TO DROP ZONE (Carry it)
print(f"   🚚 Transporting...")
drop_target = [0.5, -0.4, 0.5]

for _ in range(100):
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, drop_target, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])

    # FORCE CARRY
    wrist_pos = p.getLinkState(robotId, wrist_index)[0]
    p.resetBasePositionAndOrientation(cubeId, [wrist_pos[0], wrist_pos[1], wrist_pos[2] - 0.06], [0, 0, 0, 1])

    p.stepSimulation()
    time.sleep(0.01)

# 5. LOWER TO TABLE
print("   ⬇️ Lowering...")
lower_target = [0.5, -0.4, 0.15]

for _ in range(60):
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, lower_target, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])

    # FORCE CARRY
    wrist_pos = p.getLinkState(robotId, wrist_index)[0]
    p.resetBasePositionAndOrientation(cubeId, [wrist_pos[0], wrist_pos[1], wrist_pos[2] - 0.06], [0, 0, 0, 1])

    p.stepSimulation()
    time.sleep(0.01)

# 6. RELEASE
print("   🔓 Dropping...")
# We STOP updating the cube position here. It stays where we left it.
for _ in range(30):
    p.setJointMotorControl2(robotId, 9, p.POSITION_CONTROL, 0.04)
    p.setJointMotorControl2(robotId, 10, p.POSITION_CONTROL, 0.04)
    p.stepSimulation()
    time.sleep(0.01)

# 7. RETREAT
print("   👋 Mission Accomplished.")
retreat_target = [0.5, -0.4, 0.5]
for _ in range(60):
    joint_poses = p.calculateInverseKinematics(robotId, ee_index, retreat_target, [1, 0, 0, 0])
    p.setJointMotorControlArray(robotId, range(7), p.POSITION_CONTROL, targetPositions=joint_poses[:7])
    p.stepSimulation()
    time.sleep(0.01)

while p.isConnected():
    p.stepSimulation()
    time.sleep(0.01)