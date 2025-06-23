# Nix builder
FROM nixos/nix:2.29.0 AS builder

USER root

# install packages
ENV NIX_CONFIG="experimental-features = nix-command flakes"
COPY flake.nix /tmp/build/flake.nix
COPY flake.lock /tmp/build/flake.lock
WORKDIR /tmp/build
RUN nix profile install --profile /nix-profile

# save the nix store closure for the profile (i.e. exactly the necessary packages + profile, which
# contains a convenient collection of links to those packages)
RUN mkdir /tmp/nix-store-closure
RUN cp -R $(nix-store -qR /nix-profile) /tmp/nix-store-closure

# switch to micromamba image for final image
FROM mambaorg/micromamba:1.5.8-noble
USER root

# take along the packages that nix installed
COPY --from=builder /tmp/nix-store-closure /nix/store
COPY --from=builder /nix-profile /nix-profile
ENV PATH="/nix-profile/bin:${PATH}"

# add the code
ARG EXPERIMENT_CODE_PATH
ADD code $EXPERIMENT_CODE_PATH

# set up the python environment
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
