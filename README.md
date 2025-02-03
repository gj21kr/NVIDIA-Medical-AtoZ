# NVIDIA-Medical-AtoZ
## MONAI vs. MONAI Deploy vs. Holoscan

This document provides a comparison of **MONAI, MONAI Deploy, and Holoscan** in terms of **development, deployment, and application**.
The study notes for this repository are uploaded on my personal Notion. [Park, Jung-eun - IGX & Holoscan](https://je-park.notion.site/IGX-Holoscan-18e6df181b5b802d9f8ff15c7e4a19a5) [Park, Jung-eun - MONAI Deploy](https://je-park.notion.site/MONAI-AI-18e6df181b5b80ad8b9fe2e35f7b36e3)

## 📝 Comparison Table

| Category             | **MONAI**                                     | **MONAI Deploy**                              | **Holoscan**                                  |
|---------------------|-------------------------------------------|------------------------------------------|------------------------------------------|
| **Description**     | PyTorch-based library for medical AI model development | Framework for deploying and integrating medical AI applications | NVIDIA’s framework for real-time medical AI processing and streaming |
| **Primary Purpose** | Training and evaluation of medical AI models | Deployment and execution of medical AI models | Real-time AI processing for medical applications |
| **Key Components**  | `monai.networks`, `monai.transforms`, `monai.losses`, etc. | `MONAI Deploy App SDK`, `MONAI Inference Service`, `MONAI Deploy Operator` | `Holoscan SDK`, `GXF`, `TensorRT`, `DeepStream`, etc. |
| **Primary Use Cases** | Training medical AI models (e.g., segmentation, classification, registration) | Deploying AI models in hospitals, cloud, and edge environments | Real-time AI processing for endoscopy, ultrasound, robotic surgery, etc. |
| **Target Users** | Data scientists, medical researchers, AI developers | MLOps engineers, healthcare IT administrators | Medical device manufacturers, Edge AI developers |
| **Model Training Support** | ✅ (Supports PyTorch-based training and evaluation) | ❌ (No training functionality) | ❌ (Holoscan does not handle model training) |
| **Deployment Support** | ❌ (No direct model deployment capabilities) | ✅ (Supports Docker containers and Kubernetes) | ✅ (Optimized for medical devices and edge AI systems) |
| **Real-Time Processing** | ❌ (Batch-based processing) | ✅ (Supports inference service) | ✅ (Low-latency, real-time AI inference) |
| **Cloud Compatibility** | ✅ (Supports both cloud and local training) | ✅ (Can deploy AI services in the cloud) | ✅ (Supports both cloud and edge devices) |
| **Hardware Optimization** | ❌ (Supports CPU/GPU but lacks full optimization) | ✅ (Optimized for GPUs and NVIDIA NGC) | ✅ (Optimized for NVIDIA Jetson, Orin, A100, RTX GPUs) |
| **Framework Support** | PyTorch, TorchVision, MONAI Custom Networks | MONAI App SDK, Triton Inference Server | TensorRT, DeepStream, CUDA, GXF |
| **Medical Standard Support** | ✅ (Supports DICOM, NIfTI) | ✅ (Compatible with DICOM, FHIR, OpenVINO) | ✅ (Supports DICOM, RTSP, sensor streaming) |
| **Real-World Applications** | Brain tumor MRI segmentation, lung CT lesion detection, etc. | Integrating AI models with PACS and medical IT systems | AI-assisted ultrasound streaming, surgical AI assistance |
| **Deployment Environments** | Research and development (Jupyter Notebook, PyTorch) | On-premise, cloud, healthcare IT infrastructure | Edge AI devices, medical equipment, NVIDIA Jetson |

---

## 🔍 Summary

- **MONAI**: Best suited for developing and training medical AI models.
- **MONAI Deploy**: Designed for deploying trained AI models into real-world medical environments.
- **Holoscan**: Optimized for real-time AI processing in medical imaging and edge AI systems.

---

## 🚀 Recommended Use Cases

- **If your goal is AI model development** → Use `MONAI`
- **If your goal is AI deployment in a hospital IT system** → Use `MONAI Deploy`
- **If you need real-time medical data processing** → Use `Holoscan`

---

## 📌 Additional Resources

- [MONAI Documentation](https://monai.io/)
- [MONAI Deploy GitHub](https://github.com/Project-MONAI/monai-deploy)
- [NVIDIA Holoscan](https://developer.nvidia.com/holoscan-sdk)


