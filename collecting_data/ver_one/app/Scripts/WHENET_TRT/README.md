# convert pretrained WHENet to tensorrt

**Note**:

- To reproduce steps converting WHENet to tensorrt, use `develop` branch
- To use only, use `master branch`, it much more lighter than the `develop` branch

## 1. Setup (for usage only)

This repos require tensorrt python. There are ways to install it:

- If your system is using CUDA 11.1, go ahead and use  `pip install -r requirements.txt`
- If not using CUDA 11.1. Follow steps produce in <http://192.168.1.53:3000/dat/markdown/src/branch/master/cuda.md#user-content-6-tensorrt-installation-for-python>  
    Then do `pip install onnx 'pycuda>=2019.1.1'`

## 2. Usage only

build a plan file appropirate to your system

```bash
python3 save_to_plan.py
```

then use

```bash
python3 main.py
```

## 3. Develop - How to reproduce the convertion

workflow: `keras (tensorflow backend) -> onnx -> tensorrt`

guide: <https://developer.nvidia.com/blog/speeding-up-deep-learning-inference-using-tensorflow-onnx-and-tensorrt/>

dependencies are not included here

### steps

1. Clone `develop` branch

    ```bash
    git clone --single-branch --branch=develop http://192.168.1.53:3000/dat/WHENET_TRT.git
    ```

2. Load model using provided script from the origin repos, the re-save it using `tf.saved_model.save(model, 'saved_model')`. This is because tf 2.0 changes the way to save model

    ```bash
    python3 save_model.py
    ```

3. Convert `saved_model` to ONNX

    ```bash
    python3 -m tf2onnx.convert --saved-model saved_model --output saved_model.onnx
    ```

    Optional: test converted ONNX model

    ```bash
    python3 test_onnx.py
    ```

4. Convert ONNX to tensorrt

    ```bash
    python3 save_to_plan.py
    ```

5. Finally, do the inference and compare the result with origin repos

    ```bash
    python3 main.py
    ```
