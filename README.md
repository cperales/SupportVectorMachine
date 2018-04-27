# Support Vector Machine

This repository is a simple Python implementation of SVM, using `cvxopt`
as base solver.

- [x] Linear SVM for 2 classes
- [ ] Kernel SVM for 2 classes
- [ ] Multi classification


## Example

`svm.py` works as an entry point. Just run

```bash
python svm.py
```


## Python environment

It's recommended to install a virtual environment

```bash
virtualenv -p python3 env
```

For activating the virtual environment,

```bash
source env/bin/activate
```

To deactivate, just run ```deactivate```.

Then, you need to install the requirements

```bash
pip install -r requirements.txt
```

You can install just the requirements if you don't want to create a
virtual environment.
