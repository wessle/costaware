FROM ubuntu:18.04
RUN apt-get update && apt-get install -y python3 python3-pip
RUN pip3 install numpy torch pyyaml gym scipy wesutils
RUN useradd -ms /bin/bash costaware
USER costaware
ENV TERM xterm-256color
CMD cd && /bin/bash -l
