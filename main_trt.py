import os
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import cv2
import pyzed.sl as sl

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

def load_engine(engine_file_path):
    TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
    with open(engine_file_path, 'rb') as f:
        runtime = trt.Runtime(TRT_LOGGER)
        return runtime.deserialize_cuda_engine(f.read())

def allocate_buffers(engine):
    print("Allocating buffers...")
    input_name = engine.get_tensor_name(0)
    output_name = engine.get_tensor_name(1)
    input_shape = engine.get_tensor_shape(input_name)
    output_shape = engine.get_tensor_shape(output_name)

    h_input = cuda.pagelocked_empty(trt.volume(input_shape), dtype=np.float32)
    h_output = cuda.pagelocked_empty(trt.volume(output_shape), dtype=np.float32)
    d_input = cuda.mem_alloc(h_input.nbytes)
    d_output = cuda.mem_alloc(h_output.nbytes)
    stream = cuda.Stream()
    print("Buffers allocated successfully")
    return h_input, h_output, d_input, d_output, stream

def preprocess_image(image):
    input_image_resized = cv2.resize(image, (640, 640))
    input_image_resized = input_image_resized.transpose((2, 0, 1))  # HWC to CHW
    input_image_resized = np.ascontiguousarray(input_image_resized, dtype=np.float32)
    input_image_resized /= 255.0  # Normalize to [0, 1]
    return input_image_resized.ravel()

def do_inference(engine, context, h_input, d_input, h_output, d_output, stream, input_image):
    print("Copying data to device...")
    cuda.memcpy_htod_async(d_input, h_input, stream)
    print("Executing inference...")
    context.execute_async_v3(stream_handle=stream.handle)
    print("Copying data from device...")
    cuda.memcpy_dtoh_async(h_output, d_output, stream)
    stream.synchronize()
    return h_output

# Load the TensorRT engine
engine = load_engine('yolov8l-seg.trt')
context = engine.create_execution_context()

# Allocate buffers
h_input, h_output, d_input, d_output, stream = allocate_buffers(engine)

def read_config(file_path):
    config = {}
    with open(file_path, 'r') as file:
        for line in file:
            key, value = line.strip().split('=')
            config[key] = value
    return config

def main():
    config = read_config('config.txt')
    svo_file_path = config['SVO_VIDEOS_FILE_PATH']

    init_params = sl.InitParameters()
    init_params.set_from_svo_file(svo_file_path)
    init_params.svo_real_time_mode = True

    zed = sl.Camera()
    status = zed.open(init_params)
    if status != sl.ERROR_CODE.SUCCESS:
        print(repr(status))
        exit()

    left_mat = sl.Mat()
    right_mat = sl.Mat()

    h_input, h_output, d_input, d_output, stream = allocate_buffers(engine)
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            zed.retrieve_image(left_mat, sl.VIEW.LEFT)
            zed.retrieve_image(right_mat, sl.VIEW.RIGHT)

            left_frame = left_mat.get_data()
            right_frame = right_mat.get_data()

            left_input = preprocess_image(left_frame)
            right_input = preprocess_image(right_frame)

            # Sprawdź, czy trzeba przydzielić bufory wejściowe
            if h_input.nbytes != left_input.nbytes:
                h_input = cuda.pagelocked_empty(left_input.shape, dtype=np.float32)
                d_input = cuda.mem_alloc(h_input.nbytes)

            np.copyto(h_input, left_input)

            # Wykonaj inferencję dla lewego kadru
            left_output = do_inference(engine, context, h_input, d_input, h_output, d_output, stream, left_input)
            left_output_image = left_output.reshape((3, 640, 640)).transpose((1, 2, 0))
            left_output_image = (left_output_image * 255).astype(np.uint8)

            # Sprawdź, czy trzeba przydzielić bufory wejściowe
            if h_input.nbytes != right_input.nbytes:
                h_input = cuda.pagelocked_empty(right_input.shape, dtype=np.float32)
                d_input = cuda.mem_alloc(h_input.nbytes)

            np.copyto(h_input, right_input)

            # Wykonaj inferencję dla prawego kadru
            right_output = do_inference(engine, context, h_input, d_input, h_output, d_output, stream, right_input)
            right_output_image = right_output.reshape((3, 640, 640)).transpose((1, 2, 0))
            right_output_image = (right_output_image * 255).astype(np.uint8)

            combined_frame = cv2.hconcat([left_output_image, right_output_image])
            cv2.imshow("ZED | Segmented and Inpainted", combined_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    zed.close()

if __name__ == "__main__":
    main()
