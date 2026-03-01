#!/usr/bin/env bash
# Run TextBugger with TF JIT disabled (avoids EncoderDNN/Sqrt graph error).
# Set env var before Python starts so TensorFlow never enables JIT.
export TF_XLA_FLAGS="--tf_xla_auto_jit=-1"
exec python test_qwen_smishing.py "$@"
