# Frame identity in sensor/ROS time. Never subtract it from steady_ns.
int64 scan_end_ns
uint8 CALLBACK=0
uint8 PREPROCESSED=1
uint8 SYNCHRONIZED=2
uint8 ENQUEUED=3
uint8 DEQUEUED=4
uint8 PUBLISHED=5
uint8 stage
# NX steady_clock timestamp at the named boundary (before publish for PUBLISHED).
int64 steady_ns
