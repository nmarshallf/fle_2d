# fle_2d

Installing using pip

```rb
# install dependencies
pip install numpy scipy finufft joblib

# install package
pip install fle-2d

# download tests folder (optional)
svn checkout https://github.com/nmarshallf/fle_2d/trunk/tests
cd tests
# (alternatively) if svn is not installed, you can:
# download the project as a zip file, unzip, and navigate to the tests folder

# run test code (optional)
python3 test_fle_2d.py
```


If you find the code useful, please cite the corresponding paper:

Nicholas F. Marshall, Oscar Mickelin, Amit Singer. Fast expansion into harmonics on the disk: a steerable basis with fast radial convolutions. arXiv (2022). 
https://arxiv.org/abs/2207.13674

# Acknowledgements
We thank Yunpeng Shi for contributing a vectorized version of the code for tensor inputs consisting of multiples images.
