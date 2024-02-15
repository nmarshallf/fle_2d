# fle_2d

Installing using pip

```bash
# install dependencies
pip install numpy scipy finufft joblib

# install package
pip install fle-2d

# download tests folder (optional)
svn checkout https://github.com/nmarshallf/fle_2d/trunk/tests
cd tests
# (alternatively) if svn is not installed, you can:
# download the project as a zip file, unzip, and navigate to the tests folder.

# run test code (optional)
python3 test_fle_2d.py
```


If you find the code useful, please cite the corresponding paper:

Nicholas F. Marshall, Oscar Mickelin, and Amit Singer. Fast expansion into harmonics on the disk: A steerable basis with fast radial convolutions. SIAM Journal on Scientific Computing, 45(5):A2431–A2457, 2023.

```text
@article{marshall2023fast,
  author = {Marshall, Nicholas F. and Mickelin, Oscar and Singer, Amit},
  title = {Fast Expansion into Harmonics on the Disk: A Steerable Basis with Fast Radial Convolutions},
  journal = {SIAM Journal on Scientific Computing},
  volume = {45},
  number = {5},
  pages = {A2431-A2457},
  year = {2023},
  doi = {10.1137/22M1542775},
}
```

# Acknowledgements
We thank Yunpeng Shi for contributing a vectorized version of the code for tensor inputs consisting of multiples images.
