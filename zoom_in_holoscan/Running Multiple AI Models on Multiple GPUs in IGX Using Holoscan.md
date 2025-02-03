# 5.2 IGX와 Holoscan을 활용한 AI 가속 - Multi-GPUs 활용

If you want to **run multiple AI models simultaneously** on **multiple GPUs** in **NVIDIA IGX using Holoscan**, you need to **efficiently distribute AI inference workloads** across the available GPUs. Holoscan integrates with **NVIDIA Triton Inference Server** and **TensorRT** to manage **multi-GPU inference execution**.

---

# **📌 Overview of Multi-GPU AI Inference in Holoscan**

### **Goal**

✅ Run **multiple AI models** on **separate GPUs** using Holoscan.

✅ Ensure **parallel execution** of different inference tasks.

✅ Use **Triton or TensorRT** to handle GPU allocation efficiently.

### **Approach**

🔹 **Method 1: Triton Inference Server for Multi-GPU Execution**

➡ **Recommended for large-scale AI deployments with multiple models.**

➡ Runs multiple models **as inference services** with **automatic GPU allocation**.

🔹 **Method 2: Direct TensorRT Execution on Multiple GPUs**

➡ **Recommended for lightweight inference with low-latency requirements.**

➡ Assigns **specific models** to **specific GPUs** within Holoscan’s inference operators.

---

# **🔹 Method 1: Using Triton Inference Server for Multi-GPU AI Inference**

### **📝 How Triton Manages Multi-GPU Execution**

- **Each AI model is assigned to a separate GPU.**
- **Triton automatically batches & schedules inference requests.**
- Supports **ONNX, TensorRT, PyTorch, TensorFlow models**.
- Allows **multiple models to be served concurrently**.

---

### **📌 Step 1: Configuring Triton to Run Multiple AI Models on Different GPUs**

Each **AI model must be assigned to a specific GPU** in the Triton **model configuration file (`config.pbtxt`)**.

📂 **Directory Structure for Multi-GPU Model Deployment**

```
/models/
│── unet_model/
│   ├── 1/
│   │   ├── model.trt
│   ├── config.pbtxt
│── resnet_model/
│   ├── 1/
│   │   ├── model.onnx
│   ├── config.pbtxt
│── densenet_model/
│   ├── 1/
│   │   ├── model.pt
│   ├── config.pbtxt

```

---

### **📌 Step 2: Assign AI Models to GPUs in Triton**

Modify each model’s **config.pbtxt** to assign a **specific GPU**.

### **Example: Assigning U-Net Model to GPU 0**

📄 **`/models/unet_model/config.pbtxt`**

```yaml
name: "unet_model"
platform: "tensorrt_plan"
max_batch_size: 8
instance_group {
  kind: KIND_GPU
  count: 1
  gpus: [0]   # Assign to GPU 0
}

```

### **Example: Assigning ResNet Model to GPU 1**

📄 **`/models/resnet_model/config.pbtxt`**

```yaml
name: "resnet_model"
platform: "onnxruntime_onnx"
max_batch_size: 16
instance_group {
  kind: KIND_GPU
  count: 1
  gpus: [1]   # Assign to GPU 1
}

```

### **Example: Assigning DenseNet Model to GPU 2**

📄 **`/models/densenet_model/config.pbtxt`**

```yaml
name: "densenet_model"
platform: "pytorch_libtorch"
max_batch_size: 32
instance_group {
  kind: KIND_GPU
  count: 1
  gpus: [2]   # Assign to GPU 2
}

```

---

### **📌 Step 3: Run Triton Inference Server with Multi-GPU AI Models**

```bash
tritonserver --model-repository=/models --log-verbose=1

```

🚀 **Triton will now run multiple AI models on different GPUs in parallel!**

---

### **📌 Step 4: Run Holoscan and Connect to Triton**

Now, configure **Holoscan** to send AI inference requests to **Triton**.

📄 **`holoscan_multi_gpu.yaml`**

```yaml
gxf_application:
  - name: "multi_gpu_holoscan_pipeline"
    components:
      - name: "unet_inference"
        type: "holoscan::TritonInferenceOp"
        parameters:
          model_name: "unet_model"
          server_url: "localhost:8000"
      - name: "resnet_inference"
        type: "holoscan::TritonInferenceOp"
        parameters:
          model_name: "resnet_model"
          server_url: "localhost:8000"
      - name: "densenet_inference"
        type: "holoscan::TritonInferenceOp"
        parameters:
          model_name: "densenet_model"
          server_url: "localhost:8000"

```

### **📌 Step 5: Run Holoscan with Multi-GPU Execution**

```bash
holoscan run --config holoscan_multi_gpu.yaml

```

🚀 **Holoscan will now distribute AI inference across multiple GPUs via Triton.**

---

# **🔹 Method 2: Running TensorRT AI Inference on Multiple GPUs**

If you **don’t want to use Triton**, you can assign AI inference directly to **specific GPUs inside Holoscan inference operators**.

### **📌 Step 1: Assign AI Models to GPUs in Holoscan C++ Code**

Modify **each AI inference operator** to specify **which GPU to use**.

📄 **`unet_inference.cpp` (GPU 0)**

```cpp
#include <holoscan/holoscan.hpp>
#include <tensorrt/NvInfer.h>

class UnetInferenceOp : public holoscan::Operator {
public:
    void compute(holoscan::Context &context) override {
        cudaSetDevice(0);  // Assign to GPU 0
        auto input_tensor = context.input("image_data")->receive();

        TensorRTEngine engine("unet_model.trt");
        auto output = engine.infer(input_tensor);

        context.output("segmentation_result")->emit(output);
    }
};

```

📄 **`resnet_inference.cpp` (GPU 1)**

```cpp
class ResNetInferenceOp : public holoscan::Operator {
public:
    void compute(holoscan::Context &context) override {
        cudaSetDevice(1);  // Assign to GPU 1
        auto input_tensor = context.input("image_data")->receive();

        TensorRTEngine engine("resnet_model.trt");
        auto output = engine.infer(input_tensor);

        context.output("classification_result")->emit(output);
    }
};

```

📄 **`densenet_inference.cpp` (GPU 2)**

```cpp
class DenseNetInferenceOp : public holoscan::Operator {
public:
    void compute(holoscan::Context &context) override {
        cudaSetDevice(2);  // Assign to GPU 2
        auto input_tensor = context.input("image_data")->receive();

        TensorRTEngine engine("densenet_model.trt");
        auto output = engine.infer(input_tensor);

        context.output("feature_result")->emit(output);
    }
};

```

---

### **📌 Step 2: Configure Holoscan to Run All AI Models in Parallel**

📄 **`holoscan_multi_gpu.cpp`**

```cpp
#include <holoscan/holoscan.hpp>

class HoloscanMultiGPU : public holoscan::Application {
public:
    void compose() override {
        auto unet = make_operator<UnetInferenceOp>("UnetInference");
        auto resnet = make_operator<ResNetInferenceOp>("ResNetInference");
        auto densenet = make_operator<DenseNetInferenceOp>("DenseNetInference");

        add_flow(unet, resnet);
        add_flow(resnet, densenet);
    }
};

int main() {
    HoloscanMultiGPU app;
    app.run();
    return 0;
}

```

---

### **📌 Step 3: Compile and Run Holoscan with Multi-GPU TensorRT Execution**

```bash
mkdir build && cd build
cmake ..
make
./holoscan_multi_gpu

```

🚀 **Now, TensorRT will run each AI model on a separate GPU inside Holoscan!**

---

# **🎯 Conclusion**

✅ **Triton Inference Server** is the **best solution** for **scaling multiple AI models on multiple GPUs automatically**.

✅ **Direct TensorRT execution** gives **low-latency control over GPU execution inside Holoscan.**

✅ **Holoscan integrates both approaches seamlessly** for **high-performance AI inference** in medical imaging and robotics.

If you're working on **real-time AI-assisted surgery, robotic AI, or multi-sensor fusion**, this **multi-GPU Holoscan setup will maximize performance**! 🚀🔥

---

# **🔗 Additional Resources**

- 📖 [NVIDIA Triton Inference Server](https://developer.nvidia.com/nvidia-triton-inference-server)
- 🏥 [NVIDIA TensorRT](https://developer.nvidia.com/tensorrt)
- 🔗 [Holoscan SDK](https://developer.nvidia.com/holoscan-sdk)

Now you can **efficiently run multiple AI models on multiple GPUs using Holoscan**! 🚀😊