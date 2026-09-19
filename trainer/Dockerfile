FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

RUN apt-get update \
  && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends openssh-server \
  && rm -rf /var/lib/apt/lists/* \
  && python -m pip install --no-cache-dir "cactus-needle[train,gpu]==2.0.14"

RUN mkdir -p /opt/swapai /run/sshd \
  && printf '%s\n' \
    'PasswordAuthentication no' \
    'PermitRootLogin prohibit-password' \
    'PubkeyAuthentication yes' \
    'AllowTcpForwarding no' \
    'X11Forwarding no' \
    >> /etc/ssh/sshd_config

COPY train.py /opt/swapai/train.py
COPY entrypoint.sh /opt/swapai/entrypoint.sh
RUN chmod 755 /opt/swapai/train.py /opt/swapai/entrypoint.sh

EXPOSE 22
ENTRYPOINT ["/opt/swapai/entrypoint.sh"]
