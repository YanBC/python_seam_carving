FROM ubuntu:18.04
ARG USERID
ENV DEBIAN_FRONTEND=noninteractive

COPY ./CN.sources.list /etc/apt/sources.list

RUN apt-get update && \
    apt-get install -y \
        python3 \
        python3-dev \
        python3-pip \
        python3-opencv \
        sudo && \
    rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/bin/python3 /usr/bin/python

WORKDIR /seam_carving
COPY ./requirements.txt .
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install -r requirements.txt

RUN useradd -m -s /bin/bash -u $USERID -G sudo carver && \
    echo "carver:carver" | chpasswd

COPY . .

RUN chown carver:carver -R /seam_carving
USER carver
CMD ["python", "web_ui.py"]
