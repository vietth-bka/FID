# laymau simplified

## Setup

Note: should use cv2 with gstreamer supported in order to archieve realtime images. It usually python3.6 in ubuntu18 system.
If cv2 doesn't support gstreamer, the app will say "cv2 is built without gstreamer", and lagging is expected.

```bash
python3.6 -m pip install -r requirements.txt
sudo apt install python3-numpy
```

check if `cuda` is available

```bash
python3 -c "import torch; print(torch.cuda.current_device())"
```

## Usage

- Run the script

  ```bash
  python3 src/app.py
  ```

- Enter number of person will be collected, e.g. `2`
- Use the mouse to select a region contain exactly those `2` people mentioned above.
- The faces of those people will be saved in `save_image` directory
- Change the folder name. E.g change `ID_0` to `Nguyen Van A_example@example.com`
- Export `pkl` file by running 

  ```bash
  python3 src/export_pkl.py
  ```

- Finally update database

  ```bash
  python3 src/update_pkl.py
  ```