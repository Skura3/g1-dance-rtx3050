import warp as wp

@wp.kernel
def increment(values: wp.array(dtype=wp.float32)):
    i=wp.tid()
    values[i]=values[i]+1.0

