# matlib

This is a repository for matlib, a Matrix Library for RISC-V Accelerators. The library exposes a small number of matrix operations that can later be translated into accelerated instructions. Follow the following steps to build the library.

```
git clone git@github.com:wid4soe/matlib.git
mkdir matlib/build
cd build
cmake -DUSE_RVV=OFF -DUSE_CPU=OFF -DUSE_RVVU=ON -DCHECKSUM=OFF ..
make
```

Finally, to test using spike, use the following command. If everything goes well, there should be a list of available functions followed by a pass and the number cycles. 

```
spike --isa=rv64gcv_zicntr --varch=vlen:512,elen:32 test_matlib_rvv
```
