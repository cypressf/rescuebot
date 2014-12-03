# Rescue Bot

This repo will contain the code for a robot that searches for and locates objects of interest in a room.

## To Run

```
rosrun rescuebot runner.py
```

## To run hector exploration

```
roscore
roslaunch rescuebot neato_simulator.launch
rviz
roslaunch neato_2dnav hector_mapping_neato.launch
rosrun hector_costmap hector_costmap
roslaunch hector_exploration_node exploration_planner.launch
rosrun teleop_twist_keyboard teleop_twist_keyboard.py
rosrun hector_exploration_controller simple_exploration_controller
```

Now add /hector_exploration_node/global_costmap/costmap topic to rviz so you can see what hector exploration is thinking.