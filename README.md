# UR3e ROS2 Control and MoveIt2 Demo

Launch URSim if you don't have a real robot:
```bash
ros2 run ur_client_library start_ursim.sh -m ur3e
```

## Starting UR Driver

Launch UR driver using FAKE robot:
```bash
ros2 launch robot_control start.launch.py launch_rviz:=true use_fake_hardware:=true show_object:=true
```

Launch UR driver using REAL robot or UR Sim robot (ip=192.168.56.101):
```bash
ros2 launch robot_control start.launch.py launch_rviz:=true robot_ip:=169.254.148.159 kinematics_params_file:=$HOME/my_robot_calibration.yaml show_object:=true
```

## MoveIt
Launch MoveIt2:
```bash
ros2 launch robot_moveit_config robot_moveit.launch.py
```

Launch MoveIt2 planning GUI:
```bash
ros2 launch robot_moveit_config moveit_rviz.launch.py
```

Demo:
```bash
ros2 run robot_moveit waypoints
```

## Force/Torque Signals Visualization
Open plotjuggler for signal visualization:
```bash
ros2 run plotjuggler plotjuggler
```

Transform F/T frame:
```bash
ros2 run force_torque_sensor_broadcaster wrench_transformer_node --ros-args -p target_frames:="['world']" -r /fts_wrench_transformer/wrench:=/force_torque_sensor_broadcaster/wrench
```

## Object Visualization
Visualize the part in RViz:
```bash
ros2 run rviz_mesh_publisher publisher
```

# MeshGraphNet Integration:
1. Create Python virtual environment and install dependencies:
```bash
python3 -m venv venv
source src/planner/venv/bin/activate
pip install -r requirements.txt
```
2. Run TaskExecutor Node:
```bash
ros2 launch robot_moveit start.launch.py
```
3. Run MeshGraphNet Node:
```bash
ros2 run planner planner.py
```

# TODO:
- [x] Implement gripper control
- [x] Check force control mode with sensor logging
- [x] Modify planner to output one wrench per grasp

# Log
- 2026-03-09: FastDDS broke down after disabling firewall. Switching to CycloneDDS (added to ~/.bashrc).
- 2026-03-13: Docker permission fix: `sudo chown root:docker /var/run/docker.sock; sudo systemctl restart docker`