FROM vllm/vllm-openai:v0.14.1

RUN python3 -m pip install --no-cache-dir \
  'opentelemetry-sdk>=1.26.0,<1.27.0' \
  'opentelemetry-api>=1.26.0,<1.27.0' \
  'opentelemetry-exporter-otlp>=1.26.0,<1.27.0' \
  'opentelemetry-semantic-conventions-ai>=0.4.1,<0.5.0'

COPY docker/forceSeq /opt/vllm-plugins/forceSeq
COPY docker/vllm_entrypoint.sh /opt/vllm-plugins/vllm_entrypoint.sh

RUN chmod +x /opt/vllm-plugins/vllm_entrypoint.sh

ENV PYTHONPATH=/opt/vllm-plugins
