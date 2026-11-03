# Ollama with chembl-drug-chat:270m baked in, so the container starts with the model ready.
# The build context is the exported model folder (quantized GGUF + Modelfile):
#   docker build -f deploy/ollama.Dockerfile artifacts/<run>/ollama
# The GGUF is already quantized by export_to_ollama, so this serves the exact
# file that was registered and measured locally.
FROM ollama/ollama:0.40.13 AS build
COPY chembl-drug-chat.gguf Modelfile /model/
RUN ollama serve & \
    until ollama list >/dev/null 2>&1; do sleep 1; done; \
    ollama create chembl-drug-chat:270m -f /model/Modelfile \
 && rm -rf /usr/lib/ollama/cuda_* /usr/lib/ollama/rocm* /usr/lib/ollama/vulkan*

# CPU only: ship the binary, its CPU libs and the model store. Drops ~3.5 GB of
# GPU libraries and the raw GGUF copy. Add a GPU later = use ollama/ollama as-is.
FROM ubuntu:24.04
COPY --from=build /usr/bin/ollama /usr/bin/ollama
COPY --from=build /usr/lib/ollama /usr/lib/ollama
COPY --from=build /root/.ollama /root/.ollama
ENV OLLAMA_HOST=0.0.0.0:11434 \
    # Keep the model in memory; reloading it per request would dominate latency.
    OLLAMA_KEEP_ALIVE=-1
EXPOSE 11434
ENTRYPOINT ["/usr/bin/ollama"]
CMD ["serve"]
