# Particle filter

## Prerequisites
In order to run these codes, make sure the following requirements have been satisfied:
opencv-python>=3.4
matplotlib>=2.2
numpy>=1.14

## Running test 
```
python3 dead_reckoning.py 
python3 ParticleFilter.py
python3 map_drawing.py
```

## File description
```
dead_reckoning.py:
1.Load the raw data and implement the synchronization method.
2.Use dead-reckoning method to draw robot¡¯s trajectory and the occupancy map. 
```
```
ParticleFilter.py:
1.Load the raw data and implement the synchronization method.
2.Use particle filter method to draw robot¡¯s trajectory and the occupancy map.
3.Save the occupancy map and the robot¡¯s trajectory for further use.
```
```
Map_drawing.py:
1.Reload the occupancy map and read the images data for coloring.
2.Generate the texture map.
```
```
upload.py:
1.Reload the occupancy map and visualize for analysis.
```
## Acknowledgments
I'm grateful to all the classmates who shared their ideas on piazza.
