import os

os.environ['CUDA_MODULE_LOADING'] = 'LAZY'

import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2


def build_engine(onnx_file_path, engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(TRT_LOGGER)
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
    parser = trt.OnnxParser(network, TRT_LOGGER)

    with open(onnx_file_path, 'rb') as model:
        if not parser.parse(model.read()):
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            return None

    config = builder.create_builder_config()
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB
    config.set_flag(trt.BuilderFlag.FP16)

    profile = builder.create_optimization_profile()
    profile.set_shape(network.get_input(0).name, (1, 3, 640, 640), (1, 3, 640, 640), (1, 3, 640, 640))
    config.add_optimization_profile(profile)

    serialized_engine = builder.build_serialized_network(network, config)

    with open(engine_file_path, 'wb') as f:
        f.write(serialized_engine)

    runtime = trt.Runtime(TRT_LOGGER)
    engine = runtime.deserialize_cuda_engine(serialized_engine)
    return engine


# Build TensorRT engine
onnx_file_path = 'yolov8l-seg.onnx'
engine_file_path = 'yolov8l-seg.trt'
engine = build_engine(onnx_file_path, engine_file_path)


def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())


def allocate_buffers(engine):
    h_input = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(0)), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(engine.get_binding_shape(1)), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    return h_input, h_output, d_input, d_output, stream


def do_inference(engine, context, h_input, d_input, h_output, d_output, stream, input_image):
    cuda.memcpy_htod_async(d_input, h_input, stream)
    context.execute_async_v2(bindings=[int(d_input), int(d_output)], stream_handle=stream.handle)
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output


# Load the TensorRT engine
engine = load_engine('yolov8l-seg.trt')
context = engine.create_execution_context()

# Allocate buffers
h_input, h_output, d_input, d_output, stream = allocate_buffers(engine)

# Prepare the input image
input_image = cv2.imread('left_result.png')
input_image_resized = cv2.resize(input_image, (640, 640))
h_input = np.ravel(input_image_resized).astype(np.float32)

# Perform inference
output = do_inference(engine, context, h_input, d_input, h_output, d_output, stream, input_image_resized)
print(output)
