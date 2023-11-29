import engine as eng
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import os

def softmax(x):
    x -= np.max(x,axis=1, keepdims=True)
    a = np.exp(x)
    b = np.sum(a, axis=1, keepdims=True)
    return a/b

class Headposetrt():
    def __init__(self, max_batch_size=1, serialized_plan='saved_model.plan', binding_names='saved_model_bindings.txt'):
        assert os.path.exists(serialized_plan)
        assert os.path.exists(binding_names)
        TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
        trt_runtime = trt.Runtime(TRT_LOGGER)
        self.trt_runtime = trt_runtime
        engine, binding_names = eng.load_engine(trt_runtime, serialized_plan, binding_names)
        assert engine != None
        self.engine = engine
        self.binding_names = binding_names
        self.stream = cuda.Stream()
        self.context = engine.create_execution_context()
        self.max_batch_size = max_batch_size
        self.__mean = [0.485, 0.456, 0.406]
        self.__std = [0.229, 0.224, 0.225]
        self.__idx_tensor = np.arange(66, dtype=np.float32)
        self.__idx_tensor_yaw = np.arange(120, dtype=np.float32)
        self.__allocate_buffers(max_batch_size, trt.float32)

    def __preprocess(self, faceimg):
        """preprocess faceimg
        :param faceimg: BGR numpy array of a face, with size (112, 112)
        :returns preprocessed image
        """
        img = cv2.cvtColor(faceimg, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (224, 224))
        img = img/255
        img = (img - self.__mean) / self.__std
        return img

    def __postprocess(self, inference_result):
        """post process the network's outputs
        :param inference_result: 
        :returns yaw, pitch, roll as degree
        """
        # note that, the binding index might not in yaw, pitch, roll order
        yaw_index = [index for index, string in enumerate(self.binding_names) if 'yaw' in string][0] - 1
        pitch_index = [index for index, string in enumerate(self.binding_names) if 'pitch' in string][0] - 1
        roll_index = [index for index, string in enumerate(self.binding_names) if 'roll' in string][0] - 1
        yaw   = softmax(inference_result[yaw_index])
        pitch = softmax(inference_result[pitch_index])
        roll  = softmax(inference_result[roll_index])
        yaw   = np.sum(yaw   * self.__idx_tensor_yaw, axis=1)*3 - 180
        pitch = np.sum(pitch * self.__idx_tensor, axis=1) * 3 - 99
        roll  = np.sum(roll  * self.__idx_tensor, axis=1) * 3 - 99
        yaw = float(yaw)
        pitch = float(pitch)
        roll = float(roll)
        return (yaw, pitch, roll)

    def __allocate_buffers(self, batch_size, data_type):
        """
        This is the function to allocate buffers for input and output in the device
        Args:
            batch_size : The batch size for execution time.
            data_type: The type of the data for input and output, for example trt.float32. 

        Output:
            buffers: hold input and output buffers
            h_input: Input in the host.
            d_input: Input in the device. 
            h_output_1: Output in the host. 
            d_output_1: Output in the device. 
            h_output_2: Output in the host. 
            d_output_2: Output in the device. 
            h_output_3: Output in the host. 
            d_output_3: Output in the device. 
            stream: CUDA stream.

        """
        # since we working with a network which has 3 outputs
        assert self.engine.num_bindings == 4
        
        # Determine dimensions and create page-locked memory buffers (which won't be swapped to disk)
        # to hold host inputs/outputs
        h_input    = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(0)), dtype=trt.nptype(data_type))
        h_output_1 = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(1)), dtype=trt.nptype(data_type))
        h_output_2 = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(2)), dtype=trt.nptype(data_type))
        h_output_3 = cuda.pagelocked_empty(batch_size * trt.volume(self.engine.get_binding_shape(3)), dtype=trt.nptype(data_type))
        # Allocate device memory for inputs and outputs
        d_input    = cuda.mem_alloc(h_input.nbytes)
        d_output_1 = cuda.mem_alloc(h_output_1.nbytes)
        d_output_2 = cuda.mem_alloc(h_output_2.nbytes)
        d_output_3 = cuda.mem_alloc(h_output_3.nbytes)

        buffers = {
            'h_input'   : h_input   ,
            'h_output_1': h_output_1,
            'h_output_2': h_output_2,
            'h_output_3': h_output_3,
            'd_input'   : d_input   ,
            'd_output_1': d_output_1,
            'd_output_2': d_output_2,
            'd_output_3': d_output_3,
        }
        self.buffers = buffers
    
    def __do_inference(self, img, batch_size=1):
        """actual inference
        :param img: preprocessed numpy array
        """
        # copy img to buffer
        preprocessed = np.asarray(img).ravel()
        np.copyto(self.buffers['h_input'], preprocessed)

        # Transfer input data to the GPU
        cuda.memcpy_htod_async(self.buffers['d_input'], self.buffers['h_input'], self.stream)

        # Run inference
        self.context.execute_async(batch_size=batch_size, bindings=[int(self.buffers['d_input']), 
                                                int(self.buffers['d_output_1']), 
                                                int(self.buffers['d_output_2']), 
                                                int(self.buffers['d_output_3'])],
                                            stream_handle=self.stream.handle)

        # Transfer predictions back from GPU
        cuda.memcpy_dtoh_async(self.buffers['h_output_1'], self.buffers['d_output_1'], self.stream)
        cuda.memcpy_dtoh_async(self.buffers['h_output_2'], self.buffers['d_output_2'], self.stream)
        cuda.memcpy_dtoh_async(self.buffers['h_output_3'], self.buffers['d_output_3'], self.stream)

        # Synchronize the stream
        self.stream.synchronize()

        # Return the host output
        out = []
        out.append(self.buffers['h_output_1'].reshape((batch_size, -1)))
        out.append(self.buffers['h_output_2'].reshape((batch_size, -1)))
        out.append(self.buffers['h_output_3'].reshape((batch_size, -1)))
        return out

    def get_pose(self, faceimg):
        """get pose of an image
        :param faceimg: BGR numpy array of a face, with size (112, 112)
        :returns tupble of (yaw, pitch, roll)
        """
        img = self.__preprocess(faceimg)
        out = self.__do_inference(img)
        return self.__postprocess(out)