sudo buildctl --addr=nerdctl-container://buildkitd build \
    --frontend dockerfile.v0 \
    --opt filename=Dockerfile.pipeline_gpu \
    --local dockerfile=tools/kubeflow \
    --local context=tools/kubeflow \
    --output type=oci,name=traininghost/pipelineimage:gpu-latest | sudo nerdctl load --namespace k8s.io
