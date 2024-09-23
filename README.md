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


Then install dependency packages with pip:

```sh
pip install -r requirements.txt
```

To update the dependency packages list in requirements.txt, run:

```sh
pip freeze > requirements.txt
```


