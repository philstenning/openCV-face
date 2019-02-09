# OpenCv server

This app reads from the pi zero's video feed it then
processes the output and streams the output on its
server on the default flask port of 5000.

It assumes you already have opencv and numpy installed.

only the main.py and camera.py are used in this repos, the
other files are for trying out things.

To run:

```python3

# install the dependencies.
> pip3 install -r requirements.txt

# run the app
> python3 server.py
```
