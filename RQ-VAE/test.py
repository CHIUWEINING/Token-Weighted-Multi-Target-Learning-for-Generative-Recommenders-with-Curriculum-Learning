import numpy, importlib.metadata as md, sys
print("NumPy :", numpy.__version__)
try:
    print("k-means-constrained :", md.version("k-means-constrained"))
except md.PackageNotFoundError:
    print("k-means-constrained :  (not installed)")
