# CPU-only image. For CUDA acceleration build cuda.Dockerfile instead.
ARG BASE="python:3.11-slim"
FROM $BASE

ARG USERID="1000"
ENV DEBIAN_FRONTEND=noninteractive

# the opencv-python wheel links against these at runtime
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /seam_carving
COPY ./requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

RUN useradd -m -s /bin/bash -u $USERID carver

COPY . .

RUN chown carver:carver -R /seam_carving
USER carver
CMD ["python", "web_ui.py"]
