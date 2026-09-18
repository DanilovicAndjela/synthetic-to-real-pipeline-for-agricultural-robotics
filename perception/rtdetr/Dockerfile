FROM nvcr.io/nvidia/tao/tao-toolkit:6.25.10-pyt

WORKDIR /workspace

# PEP 668: this image's system Python is externally managed
RUN pip install --no-cache-dir --break-system-packages \
    "protobuf<6" \
    google-cloud-storage

RUN mkdir -p /workspace/weights && \
    wget -O /workspace/weights/resnet50-0676ba61.pth \
    https://download.pytorch.org/models/resnet50-0676ba61.pth
    
RUN wget -O /workspace/weights/rtdetr_pretrained.pth \
    "https://api.ngc.nvidia.com/v2/models/nvidia/tao/trafficcamnet_transformer_lite/versions/trainable_resnet50_v2.0/files/resnet50_trafficcamnet_rtdetr.pth" && \
    echo "9e21450a1eac2012ab713dc103e1655eb438a73a125e2f23b0a2ba8c0583ea6a  /workspace/weights/rtdetr_pretrained.pth" \
    | sha256sum -c -

COPY train.py /workspace/train.py
COPY specs/ /workspace/specs/

ENTRYPOINT ["python", "-u", "train.py"]

