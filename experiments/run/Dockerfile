FROM mambaorg/micromamba:1.5.8-noble

# prepare installing packages
USER root
RUN apt-get update -q

# install latex
ARG TL_MIRROR="https://mirror.ctan.org/systems/texlive/tlnet"

RUN apt-get install perl curl wget fontconfig ca-certificates gnupg -q -y --no-install-recommends && \
    mkdir "/tmp/texlive" && cd "/tmp/texlive" && \
    wget "$TL_MIRROR/install-tl-unx.tar.gz" && \
    tar xzvf ./install-tl-unx.tar.gz && \
    ( \
        echo "selected_scheme scheme-minimal" && \
        echo "instopt_adjustpath 0" && \
        echo "tlpdbopt_install_docfiles 0" && \
        echo "tlpdbopt_install_srcfiles 0" && \
        echo "TEXDIR /opt/texlive/" && \
        echo "TEXMFLOCAL /opt/texlive/texmf-local" && \
        echo "TEXMFSYSCONFIG /opt/texlive/texmf-config" && \
        echo "TEXMFSYSVAR /opt/texlive/texmf-var" && \
        echo "TEXMFHOME ~/.texmf" \
    ) > "/tmp/texlive.profile" && \
    "./install-tl-"*"/install-tl" --location "$TL_MIRROR" -profile "/tmp/texlive.profile" && \
    rm -vf "/opt/texlive/install-tl" && \
    rm -vf "/opt/texlive/install-tl.log" && \
    rm -vrf /tmp/*

ENV PATH="${PATH}:/opt/texlive/bin/x86_64-linux"

# install latex packages
RUN tlmgr install scheme-basic

# install linux packages
# original versions are given, but they cannot be used since apt repositories don't keep old package versions
# (cm-super, dvipng, texlive-latex-extra and texlive-fonts-recommended are necessary for matplotlib text rendering)
RUN apt-get install -q -y --no-install-recommends\
    # tmux=3.4-1build1 \
    tmux \
    # time=1.9-0.2build1 \
    time \
    # cm-super=0.3.4-17 \
    cm-super \
    # dvipng=1.15-1.1 \
    dvipng \
    # texlive-latex-extra=2023.20240207-1 \
    texlive-latex-extra \
    # texlive-fonts-recommended=2023.20240207-1 \
    texlive-fonts-recommended \
    # texlive-luatex=2023.20240207-1 && \
    texlive-luatex && \
    rm -rf /var/cache/apt && \
    apt-get clean

# set build argument
ARG EXPERIMENT_CODE_PATH

# add the code
ADD code $EXPERIMENT_CODE_PATH

# set up the conda environment
RUN cp -r $EXPERIMENT_CODE_PATH/. /tmp
RUN micromamba install -y --name base --file /tmp/environment.yml
RUN micromamba clean --all --yes

# set the environment variables
ENV EXPERIMENT_CODE_PATH=$EXPERIMENT_CODE_PATH
ARG EXPERIMENT_PATH
ENV EXPERIMENT_PATH=$EXPERIMENT_PATH
ARG EXPERIMENT_DEFINITIONS_PATH
ENV EXPERIMENT_DEFINITIONS_PATH=$EXPERIMENT_DEFINITIONS_PATH
ARG EXPERIMENT_DATA_PATH
ENV EXPERIMENT_DATA_PATH=$EXPERIMENT_DATA_PATH
ARG EXPERIMENT_OUTPUT_PATH
ENV EXPERIMENT_OUTPUT_PATH=$EXPERIMENT_OUTPUT_PATH
ARG EXPERIMENT_RUNNER_PATH
ENV EXPERIMENT_RUNNER_PATH=$EXPERIMENT_RUNNER_PATH
ARG EXPERIMENT_BASH_UTIL_PATH
ENV EXPERIMENT_BASH_UTIL_PATH=$EXPERIMENT_BASH_UTIL_PATH
ARG CACHE_PATH
ENV CACHE_PATH=$CACHE_PATH

# copy experiment definition
ADD definitions $EXPERIMENT_DEFINITIONS_PATH
ADD definitions/0_runner $EXPERIMENT_RUNNER_PATH
ADD definitions/0_bash_util $EXPERIMENT_BASH_UTIL_PATH

# copy data
ADD data $EXPERIMENT_DATA_PATH

CMD bash $EXPERIMENT_RUNNER_PATH/run_experiment.sh
