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
roslaunch rescuebot hector_exploration.launch
```

Now add /hector_exploration_node/global_costmap/costmap topic to rviz so you can see what hector exploration is thinking.