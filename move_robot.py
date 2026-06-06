import pybullet as p
import pybullet_data
import time
import math

# 1. SETUP SIMULATION
# -------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# Load Robot
startPos = [0, 0, 0]
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, useFixedBase=True)

# 2. IDENTIFY JOINTS
# ------------------
# The Franka Panda has many joints (fingers, wheels, etc.).
# We only care about the 7 arm joints that allow it to move.
# Their indices in PyBullet are usually: 0, 1, 2, 3, 4, 5, 6
arm_joints = [0, 1, 2, 3, 4, 5, 6]

# Reset the robot to a "Ready" pose (so it doesn't flop down)
# These are joint angles in radians
rest_poses = [0, -0.215, 0, -2.57, 0, 2.356, 2.356]
for i in range(7):
    p.resetJointState(robotId, arm_joints[i], rest_poses[i])

# 3. CONTROL LOOP (The Brain)
# ---------------------------
print("Moving the robot... (Press Ctrl+F2 to stop)")
t = 0
while True:
    p.stepSimulation()
    t += 0.01

    # Calculate a simple sine wave movement for Joint 1 (The shoulder)
    # This makes the arm wave back and forth
    target_angle = 0.5 * math.sin(t)

    # SEND COMMAND TO MOTOR
    # We tell Joint 1 to move to the 'target_angle'
    p.setJointMotorControl2(
        bodyUniqueId=robotId,
        jointIndex=1,  # Controlling the second joint
        controlMode=p.POSITION_CONTROL,
        targetPosition=target_angle,
        force=500  # Max motor strength
    )

    time.sleep(1. / 240.)