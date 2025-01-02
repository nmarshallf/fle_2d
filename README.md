# fle_2d

Installing in conda environment

```bash
conda create --name fle python=3.9 pip
conda activate fle
conda install -c conda-forge pip numpy scipy joblib matplotlib finufft
pip install fle-2d
```

Installing using pip

```bash
pip install numpy scipy joblib matplotlib finufft fle-2d
```

Testing install

```bash
git clone https://github.com/nmarshallf/fle_2d.git
# Or download folder and unzip
cd fle_2d/tests/
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


## Usage

Given an image represented by a 2D array of size LxL that you want to expand into the disk harmonic basis, first create a basis object by calling
```python
from fle_2d import FLEBasis2D
L = 128         #replace this by the side-length of your image
bandlimit = L   #maximum number of basis functions to use
eps = 1e-7      #desired accuracy
fle = FLEBasis2D(L, bandlimit, eps)
```
Here, eps is the accuracy desired in applying the basis expansion, corresponding to the epsilon in Theorem 4.1 in the paper. "Bandlimit" is a parameter that determines how many basis functions to use and corresponds to the variable lambda in equation (5.1) in the paper, scaled so that L is the maximum suggested.

All arguments to FLEBasis2D:

- L:    size of image to be expanded

- bandlimit:    bandlimit parameter (scaled so that L is max suggested)

- eps:     requested relative precision

- maxitr:      maximum number of iterations for the expand method (if not specified, pre-tuned values are used)

- maxfun:      maximum number of basis functions to use (if not specified, which is the default, the number implied by the choice of bandlimit is used)


- mode:       choose either "real" or "complex" (default) output, using either real-valued or complex-valued basis functions


    
To go from the image to the basis coefficients, you would then call either

```python
coeff = fle.evaluate_t(image)
```

which applies the operator $\tilde{B}^*$ in Theorem 4.1 of the paper, or 

```python
coeff = fle.expand(image)
```
which solves a least squares problem instead of just applying equation $\tilde{B}^*$ once. The latter can be more accurate, but takes a bit longer since it applies evaluate_t ```maxitr``` times using Richardson iteration.

Once you have coefficients ```coeff``` in the basis, you can evaluate the corresponding function with expansion coefficients ```coeff``` on the LxL grid by running

```python
image = fle.evaluate(coeff)
```

which corresponds to applying the operator $\tilde{B}$ in Theorem 4.1 in the paper.
