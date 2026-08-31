FROM nvcr.io/nvidia/tao/tao-toolkit:6.25.10-pyt

WORKDIR /workspace

# PEP 668: this image's system Python is externally managed
RUN pip install --no-cache-dir --break-system-packages \
    "protobuf<6" \
    google-cloud-storage

RUN mkdir -p /workspace/weights && \
    wget -O /workspace/weights/resnet50-0676ba61.pth \
    https://download.pytorch.org/models/resnet50-0676ba61.pth

COPY train.py /workspace/train.py
COPY specs/ /workspace/specs/

ENTRYPOINT ["python", "-u", "train.py"]

