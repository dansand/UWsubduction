FROM underworldcode/underworld2


# set working directory to /opt, and install underworld files there.
WORKDIR /opt

RUN pip3 install networkx
RUN pip3 install easydict
RUN pip3 install naturalsort
RUN pip3 install "networkx==1.11"
RUN pip3 install pint
RUN pip3 install pandas


USER root

RUN git clone https://github.com/dansand/UWsubduction.git

# change user and update pythonpath
ENV PYTHONPATH $PYTHONPATH:$UW2_DIR
#ENV PYTHONPATH /workspace/user_data/UWGeodynamics:$PYTHONPATH
ENV PYTHONPATH /opt/UWsubduction:$PYTHONPATH

# move back to workspace directory
WORKDIR /workspace


# CHANGE USER
USER $NB_USER
WORKDIR /workspace
