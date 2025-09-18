# Jet AI

The end-to-end pipeline to build, train, and fine-tune custom LLM models you own 🚀

## User Facing Features

### Training engines
- HuggingFace training engine for CPU or older GPU training for Kaggle and Colab environments
- Unsloth/DeepSpeed engine for GPU deployments on H100 instances with optimized kernels and auto-scaling enabled. Enable FlashAttention2 to run faster

### Using the platform

- Use the no-code UI to build your model and push to deploy 🚀

OR 

- Use the notebook and sdk to abstract model training using industry best-practices and SOTA techniques
- Use the CLI to start training, evaluate, and deploy with one command ✨

### Fine Tuning a model

- Pick a curated dataset (premium feature) or upload your data and fine tune your model
- Pick from many open-weight models to fine tune, ie Traditional LLMs, Mixture-of-Experts, or use our Mixture-of-Agents model

### Deployment

- Deploy to a vLLM inference endpoint for state-of-the-art throughput capacity

<hr />

## Technical features
- Unsloth
- DeepSpeed & Accelerate for multi-gpu setups
- QLoRA + Mixed Precison (FP16/8/4 & BP16/8/4) for optimization
- vLLM deployment
- HuggingFace open-weight and open-source models and fine-tuning data
- Auto-enable multi-gpu processess by detecting environment or run on single gpu
