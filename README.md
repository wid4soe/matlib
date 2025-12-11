# matlib

This is a repository for matlib, a Matrix Library for RISC-V Accelerators. The library exposes a small number of matrix operations that can later be translated into accelerated instructions. Follow the following steps to build the library. The USE_TYPE option takes float32 and float64, and the USE_RVV, USE_RVVU, and USE_RVVF are meant to be exclusive.

```
git clone git@github.com:wid4soe/matlib.git
mkdir matlib/build
cd build
cmake -DUSE_RVV=OFF -DUSE_CPU=OFF -DUSE_RVVU=ON -DUSE_RVVF=OFF -DCHECKSUM=OFF -DUSE_TYPE=float32 ..
make
```

Finally, to test using spike, use the following command. If everything goes well, there should be a list of available functions followed by a pass and the number cycles. 

```
spike --isa=rv64gcv_zv512b test_matlib_rvv
```
