`preprocess.py` creates a combined dataset combining both real and virtual events.

The `prepare` function in the C++ file creates a matrix. Each column of the matrix is a column vector of size 2. The first co-ordinate in the column vectors corresponds to `pt` and the second one corresponds to `y`. By treating them as vectors and batching them in the form of a matrix, the distance (the modfied norm) can easily be calculated.

The `DistanceAndWeight` struct groups together, the distance and the weight. So that, when distances are sorted on the basis of their proximity to the seed, the weights also get correspondingly rearranged. The weights are actually references so that any modfifications made persist throughout the program.

The scale is a column vector `[0, 10]`. When calculating norm, since each of the coordinates in the displacement vectors get squared, `10` also gets squared giving us `100`, the required scale. `%=` in `armadillo` is used for inplace element-wise multiplication.

The algorithmic complexity should probably be `O(neg)` where `neg` is the number of negative events, because of using only single non-nested `for` loops.

# Discussion Question

I suppose it's because we need both `pt` and `y` to contribute to the distance metric equally.
