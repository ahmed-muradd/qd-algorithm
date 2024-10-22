## Start developing
### venv
Create a virtual enviroment and source it:

using bash shell (linux, macos, wsl):
```sh
python3 -m venv .venv
source .venv/bin/activate
```

using CMD (Windows):
```sh
python3 -m venv .venv
.venv\Scripts\activate
```

### install depenedencies

Then install dependency packages with pip:

```sh
pip install -r requirements.txt
```

To update the dependency packages list in requirements.txt, run:

```sh
pip freeze > requirements.txt
```


Result files from qdpy is put into output folder. The output folder is ignored by git.
Result files from QDAX (not used for now) is put into logs folder. The logs folder is ignored by git.


## Using GPU acceleration

To use JAX on Nvidia GPU:
```sh
pip install -U "jax[cuda12]"
```
[read more](https://jax.readthedocs.io/en/latest/installation.html#installation)
