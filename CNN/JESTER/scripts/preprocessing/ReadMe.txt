Steps involved in dataset prep:
Use ffmpeg to convert folder of jpegs to mp4 
Use PIX2NVS to convert mp4 to NVS 
Use testing.py script to convert raw NVS data to an interleaved 2 channel 176x100 matrix saved as .dat file
Repeat for each class

Video length = 1s
Dimensions = 176x100 dimensions (interleaved as 352x100)

Parameters used: 
time window (t_r to t_ru) size: 20%
PIX2NVS parameter N = 1 (default)
fps = 37
