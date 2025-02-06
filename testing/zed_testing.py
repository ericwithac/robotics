import pyzed.sl as sl
import cv2
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import time

# TensorRT Logger
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# Load the TensorRT Engine
def load_engine(engine_path):
    with open(engine_path, "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
        return runtime.deserialize_cuda_engine(f.read())

# Allocate memory for TensorRT execution
def allocate_buffers(engine):
    inputs, outputs, bindings = [], [], []
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        bindings.append(int(device_mem))
        if engine.binding_is_input(binding):
            inputs.append({'host': host_mem, 'device': device_mem})
        else:
            outputs.append({'host': host_mem, 'device': device_mem})
    return inputs, outputs, bindings, stream

# Run object detection on an image
def detect_objects(context, inputs, outputs, bindings, stream, image):
    img_resized = cv2.resize(image, (640, 640))
    img_normalized = img_resized / 255.0
    img_transposed = np.transpose(img_normalized, (2, 0, 1)).astype(np.float32)
    np.copyto(inputs[0]['host'], img_transposed.ravel())

    start_time = time.time()
    cuda.memcpy_htod_async(inputs[0]['device'], inputs[0]['host'], stream)
    context.execute_async_v2(bindings, stream.handle, None)
    cuda.memcpy_dtoh_async(outputs[0]['host'], outputs[0]['device'], stream)
    stream.synchronize()
    end_time = time.time()

    detections = outputs[0]['host'].reshape(-1, 85)  # Format: [x, y, w, h, conf, class1, class2, ...]
    print(f"Inference Time: {end_time - start_time:.3f} sec")
    return detections

# Initialize ZED Camera
zed = sl.Camera()
init_params = sl.InitParameters(camera_resolution=sl.RESOLUTION.HD720, depth_mode=sl.DEPTH_MODE.NONE)
status = zed.open(init_params)

if status != sl.ERROR_CODE.SUCCESS:
    print("Failed to open ZED Camera")
    exit(1)

image = sl.Mat()

# Load YOLOv11 TensorRT Model
engine_path = "yolo11m.engine"
engine = load_engine(engine_path)
inputs, outputs, bindings, stream = allocate_buffers(engine)
context = engine.create_execution_context()

# Capture frames and run YOLO
while True:
    if zed.grab() == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        frame = image.get_data()[:, :, :3]  # Remove alpha channel

        detections = detect_objects(context, inputs, outputs, bindings, stream, frame)

        # Draw bounding boxes
        for det in detections:
            x, y, w, h, conf, class_id = det[:6]
            if conf > 0.5:
                x1, y1 = int(x - w / 2), int(y - h / 2)
                x2, y2 = int(x + w / 2), int(y + h / 2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Class {int(class_id)}: {conf:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        cv2.imshow("YOLOv11 ZED Camera", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

zed.close()
cv2.destroyAllWindows()