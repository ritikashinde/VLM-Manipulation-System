import pybullet as p
import pybullet_data
import time
import math
import numpy as np

# 1. SETUP SIMULATION
# -------------------
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.8)
p.loadURDF("plane.urdf")

# Load Robot
robotId = p.loadURDF("franka_panda/panda.urdf", [0, 0, 0], useFixedBase=True)


# 2. SETUP CAMERA (The "Eye")
# ---------------------------
# We place a camera 1 meter away, looking at the robot
def get_camera_image():
    # Camera position setup
    viewMatrix = p.computeViewMatrix(
        cameraEyePosition=[1.5, 0, 1.0],  # Camera is 1.5m away, 1m up
        cameraTargetPosition=[0, 0, 0.5],  # Looking at the robot's chest
        cameraUpVector=[0, 0, 1]
    )
    projectionMatrix = p.computeProjectionMatrixFOV(
        fov=60.0,
        aspect=1.0,
        nearVal=0.1,
        farVal=100.0
    )

    # Take the photo
    width, height, rgbImg, depthImg, segImg = p.getCameraImage(
        width=224, height=224,  # Standard size for AI models
        viewMatrix=viewMatrix,
        projectionMatrix=projectionMatrix,
        renderer=p.ER_BULLET_HARDWARE_OPENGL
    )
    return width, height


# 3. RUN SIMULATION
# -----------------
print("Robot is moving and 'watching' itself...")
t = 0
while True:
    p.stepSimulation()
    t += 0.05

    # Move the arm (Sine wave dance)
    target_angle = 0.5 * math.sin(t)
    p.setJointMotorControl2(robotId, 1, p.POSITION_CONTROL, target_angle)

    # Every 50 steps, take a picture
    # (In a real AI loop, we do this every step)
    if int(t * 20) % 50 == 0:
        w, h = get_camera_image()
        print(f"Snap! Captured image size: {w}x{h}")

    time.sleep(1. / 240.)