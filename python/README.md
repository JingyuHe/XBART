# XBART python package

## Installation

### From PyPi

To install XBART from PYPI use `pip install xbart`

### From source

#### linux/MacOS 

For general installation run `./build_py.sh -d`. 

If you are making changes to the C++ files in \xbart please download SWIG. 
Once installed, run `./build_py.sh -s -d`

#### PC

Use `build_py3.cmd` (requires SWIG)

# Example

The XBART API is sklearn like, for more examples refer to \tests\HousePrice and \tests\Titanic. Here is a high level overview:

```python
from xbart import XBART

xbt = XBART(num_trees = 100, num_sweeps = 40, burnin = 15)
xbt.fit(x_train,y_train)
xbart_yhat_matrix = xbt.predict(x_test)  # Return n X num_sweeps matrix
y_hat = xbart_yhat_matrix[:,15:].mean(axis=1) # Use mean a prediction estimate
```



