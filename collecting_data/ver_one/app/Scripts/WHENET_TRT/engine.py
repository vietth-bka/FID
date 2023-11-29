#%%
import tensorrt as trt

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
trt_runtime = trt.Runtime(TRT_LOGGER)

def build_engine(onnx_path, shape = [1,224,224,3]):

    """
    This is the function to create the TensorRT engine
    Args:
        onnx_path : Path to onnx_file. 
        shape : Shape of the input of the ONNX file. 

    """
    EXPLICIT_BATCH = 1
    with trt.Builder(TRT_LOGGER) as builder,\
        builder.create_network(EXPLICIT_BATCH) as network,\
        trt.OnnxParser(network, TRT_LOGGER) as parser:
        builder.max_workspace_size = (256 << 20)
        with open(onnx_path, 'rb') as model:
            if not parser.parse(model.read()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise RuntimeError()
        network.get_input(0).shape = shape
        engine = builder.build_cuda_engine(network)
        return engine

def save_engine(engine, file_name, engine_bindings):
    """serialize model to file, save binding names for inference
    """
    buf = engine.serialize()
    with open(file_name, 'wb') as f:
        f.write(buf)
    with open(engine_bindings, 'w') as f:
        for i in range(engine.num_bindings):
            f.write(engine.get_binding_name(i) + '\n')
    
def load_engine(trt_runtime, plan_path, engine_bindings):
    """deserilize cuda engine, read bidings names for inference
    """
    with open(plan_path, 'rb') as f:
        engine_data = f.read()
    
    engine = trt_runtime.deserialize_cuda_engine(engine_data)
    with open(engine_bindings, 'r') as f:
        bindings = [i.rstrip() for i in f.readlines()]
    
    return engine, bindings

# if __name__ == '__main__':
    # load_engine(trt_runtime, 'saved_model.plan', 'saved_model_bindings.txt')
    # print('Done')