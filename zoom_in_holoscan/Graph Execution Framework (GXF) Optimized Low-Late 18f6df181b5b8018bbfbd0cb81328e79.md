# Graph Execution Framework (GXF): Optimized Low-Latency Processing in NVIDIA Holoscan

## **🚀 Introduction**

The **Graph Execution Framework (GXF)** is the backbone of **NVIDIA Holoscan**, providing an **optimized execution engine** for **real-time AI applications** that require **low-latency, high-throughput processing**.

GXF enables **efficient scheduling, synchronization, and execution** of AI models and data pipelines, making it a critical component for **medical imaging, robotic-assisted surgery, and other sensor-driven applications**.

This article takes an **in-depth look at GXF, its architecture, key components, and how it enhances real-time AI performance**.

---

# **📌 What is GXF?**

GXF (**Graph Execution Framework**) is a **C++-based runtime framework** that **optimizes the execution of AI workflows** in real-time sensor-driven applications.

## **🔹 Why is GXF Needed?**

✅ **Ultra-Low Latency Execution** (Microsecond-scale AI inference).

✅ **Efficient Multi-Threaded Processing** (Parallel Execution).

✅ **Optimized for High-Speed Sensor Data Streams** (Medical Imaging, Robotics).

✅ **Supports CPU, GPU, DPU & FPGA Workloads**.

GXF ensures that **sensor data, AI models, and computational tasks** are processed **sequentially or in parallel**, depending on the application’s needs.

---

# **🔹 GXF Core Concepts & Architecture**

### **1️⃣ Entities: The Building Blocks of a GXF Graph**

An **Entity** is the fundamental unit of execution in GXF. Each **Entity** represents a **task, AI model, or data processing step**.

📌 **Examples of Entities:**

- **Data acquisition tasks** (e.g., ultrasound image capture).
- **AI inference tasks** (e.g., TensorRT AI segmentation model).
- **Post-processing tasks** (e.g., overlaying AI results on an endoscopy video).

📌 **Example: Defining an Entity in YAML**

```yaml
entities:
  - name: "sensor_data_loader"
    components:
      - name: "sensor_op"
        type: "holoscan::SensorOp"
  - name: "ai_inference"
    components:
      - name: "inference_op"
        type: "holoscan::AIInferenceOp"

```

🚀 **Each Entity is a task that can execute independently or in sequence.**

---

### **2️⃣ Components: Modular Processing Units**

A **Component** is a functional module inside an **Entity** that performs **specific actions** (e.g., acquiring sensor data, running AI inference, saving results).

📌 **Types of Components:**

| Component | Purpose |
| --- | --- |
| **holoscan::SensorOp** | Acquires real-time sensor data |
| **holoscan::AIInferenceOp** | Runs AI models on GPUs |
| **holoscan::FileWriterOp** | Saves processed data to disk |
| **holoscan::ImageDisplayOp** | Displays AI-enhanced images |

📌 **Example: AI Inference Component (YAML)**

```yaml
components:
  - name: "ai_inference_component"
    type: "holoscan::AIInferenceOp"
    parameters:
      model_path: "/models/unet.onnx"
      input_tensor: "image_data"
      output_tensor: "segmentation_result"

```

🚀 **Each Component performs a specific function inside a GXF Entity.**

---

### **3️⃣ Graphs: Connecting Entities for Execution**

A **Graph** defines the overall pipeline, connecting multiple **Entities** to create a real-time AI workflow.

📌 **How Graphs Work in GXF:**

- Entities **send and receive data** through directed connections.
- **AI inference, sensor data acquisition, and visualization** are connected as a pipeline.

📌 **Example: Defining a Graph for AI-Assisted Endoscopy**

```yaml
gxf_application:
  - name: "holoscan_pipeline"
    components:
      - name: "sensor_input"
        type: "holoscan::SensorOp"
      - name: "ai_processing"
        type: "holoscan::AIInferenceOp"
      - name: "visualization"
        type: "holoscan::ImageDisplayOp"
connections:
  - source: "sensor_input"
    target: "ai_processing"
  - source: "ai_processing"
    target: "visualization"

```

🚀 **GXF Graphs define how AI-powered workflows execute in real time.**

---

### **4️⃣ Executors: Scheduling & Running the Graph**

GXF uses an **Executor** to **schedule and run tasks** in a **high-performance, low-latency manner**.

📌 **Types of Execution Strategies:**

| Executor | Purpose |
| --- | --- |
| **Synchronized Execution** | Runs tasks sequentially (one after another) |
| **Multi-threaded Execution** | Runs tasks in parallel (high performance) |
| **GPU-accelerated Execution** | Runs AI inference on NVIDIA GPUs |

📌 **Example: Multi-Threaded Execution for High-Speed AI**

```yaml
executors:
  - name: "gpu_executor"
    type: "holoscan::ThreadPoolExecutor"
    parameters:
      num_threads: 4

```

🚀 **GXF Executors ensure AI tasks are scheduled efficiently.**

---

# **🔹 Real-World Application of GXF**

💡 **Scenario: AI-Assisted Robotic Surgery**

1. **Real-time video feed** from a surgical camera is acquired using `SensorOp`.
2. AI-powered **semantic segmentation** detects **critical tissue structures** (`AIInferenceOp`).
3. The **processed image** is **overlaid onto the surgeon's display** (`ImageDisplayOp`).
4. AI inference results are stored in **hospital PACS** for review.

📌 **GXF-Based AI-Powered Surgery Pipeline:**

```yaml
gxf_application:
  - name: "surgical_ai_pipeline"
    components:
      - name: "camera_input"
        type: "holoscan::SensorOp"
      - name: "ai_segmentation"
        type: "holoscan::AIInferenceOp"
      - name: "surgical_display"
        type: "holoscan::ImageDisplayOp"
connections:
  - source: "camera_input"
    target: "ai_segmentation"
  - source: "ai_segmentation"
    target: "surgical_display"

```

🚀 **GXF ensures real-time AI inference in robotic surgery applications.**

---

# **🔹 Deployment: Running a GXF Pipeline on NVIDIA IGX**

Holoscan applications using GXF can be deployed on **edge computing platforms** such as **NVIDIA IGX Orin** for **real-time AI inferencing**.

📌 **Deploying a GXF AI Pipeline on IGX:**

```bash
holoscan run --config configs/surgical_ai.yaml

```

📌 **Deploying on Kubernetes (Cloud/On-Premise)**

```bash
kubectl apply -f gxf_holoscan_k8s.yaml

```

🚀 **Deploy AI-powered robotic surgery applications in hospitals & cloud environments.**

---

# **🎯 Conclusion**

The **Graph Execution Framework (GXF)** is a **game-changer for real-time AI inferencing**, enabling:

✅ **Ultra-low-latency sensor processing** (AI-assisted surgery, medical imaging).

✅ **High-performance execution across CPUs, GPUs, and DPUs**.

✅ **Efficient AI model scheduling & parallel execution**.

✅ **Scalable AI-powered pipelines for real-world medical applications**.

If you’re working on **AI-powered surgery, real-time medical imaging, or AI-driven robotics**, **GXF provides the ideal framework for real-time inferencing**. 🚀🔥

---

# **🔗 Additional Resources**

- 📖 [NVIDIA Holoscan SDK Docs](https://developer.nvidia.com/holoscan-sdk)
- 🏥 [NVIDIA Clara for Healthcare](https://developer.nvidia.com/clara)
- 🔗 [Holoscan GitHub](https://github.com/nvholoscan)

Now you have a **detailed understanding of how GXF powers real-time AI workflows in NVIDIA Holoscan**! Let me know if you need more details. 🚀😊