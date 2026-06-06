import pybullet as p
import pybullet_data
import time

# 1. SETUP SIMULATION
# -------------------
# Connect to the graphical interface
physicsClient = p.connect(p.GUI)

# Add path to built-in assets (robot models, floor, etc.)
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Set gravity (Earth gravity pointing down)
p.setGravity(0, 0, -9.8)

# 2. LOAD ASSETS
# --------------
# Load the checkered floor
planeId = p.loadURDF("plane.urdf")

# Load the Franka Emika Panda robot arm
# We load it at position (0,0,0) with a fixed base so it doesn't fall over
startPos = [0, 0, 0]
robotId = p.loadURDF("franka_panda/panda.urdf", startPos, useFixedBase=True)

# 3. RUN SIMULATION
# -----------------
print("Simulation running! (Close the window to stop)")

# Run the simulation loop
for i in range(10000):
    p.stepSimulation()
    # Slow down slightly so humans can see it (1/240th of a second per step)
    time.sleep(1./240.)

p.disconnect()