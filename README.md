# SubROC Experiments

[![DOI](https://zenodo.org/badge/984705403.svg)](https://doi.org/10.5281/zenodo.16952343)

SubROC is a machine learning model evaluation method based on Subgroup Discovery in the Exceptional Model Mining framework, aimed at providing interpretable descriptions of strengths and weaknesses for a given model on structured data.

This is the code repository accompanying the paper "SubROC: AUC-Based Discovery of Exceptional Subgroup Performance for Binary Classifiers".
This code is intended to enable the reproduction of our results.
For use of our method we integrated the implementation into the [pysubgroup](https://github.com/flemmerich/pysubgroup) library for subgroup discovery.

## Quickstart Instructions

To quickly reproduce one of the experiments, perform the following steps.

1. Ensure you are working on a Linux system with Docker installed and set up in rootless mode.
2. Clone this repository and `cd` into its root directory.
3. Run the following two commands in your terminal, replacing `<experiment-name>` with the name of the experiment you would like to run.

```bash
bash experiments/run/build.sh
bash experiments/run/start.sh <experiment-name>
```

The output will appear in `./experiments/outputs/<experiment-name>`.
You can start multiple experiments in parallel by repeatedly issuing the second command with different experiment names.
See below for a legend of the experiment names and script arguments, such as for limiting computational resources of an experiment run.

## Experiment Overview

The code execution is organized into experiments.
Experiments run code inside a docker container as specified by an experiment definition.
An experiment consists of stages which are executed sequentially.
Stages are made up of steps which are executed in parallel.
Steps are defined through bash scripts.

**exp-27:** skew plots on synthetic data (Figure 3 and 4)

**exp-44 - exp-51:** result set properties (Table 1, supp. Tables 5-6)

**exp-84:** subgroup injection (Figure 2, supp. Figures 6-7)

**exp-118 - exp-149:** measuring optimistic estimate runtime speedups (Table 3, supp. Tables 13-17, supp. Figure 8)

**exp-152:** Adult case study (Table 2, supp. Tables 7-12)

*Notes:*
- Experiments using the Census KDD dataset (48, 122, 130, 138, 146) require a large amount of memory (about 170GB per parallel run, of which there are up to 12).
    - We changed the search algorithm in exp-48 to BestFirstSearch to improve memory consumption. Results might therefore be slightly different to those in the paper. For exact reproduction, change this back to Apriori in `experiments/definitions/exp-48/stage-03/base_config.yaml`.
- Sometimes the Adult dataset fails to load because OpenML is temporarily unavailable. Try again later in that case.
- The preprocessing splits the data into 4 equal parts. Only the first 3 are used for the paper results.

### Meta Reports

When results are present in `experiments/outputs`, the notebooks in `experiments/meta_reports` may be used to combine them into overview tables similar to those in the paper.

## Prerequisites for Execution

- Linux system (with terminal access)
- Docker (rootless)
- recommended: tmux (Start with `experiments/tmux_init.sh`, then run experiments inside the tmux session and use the `resource-monitor` window to observe the resource consumption.)

The experiments are intended to run in Docker [rootless mode](https://docs.docker.com/engine/security/rootless).
This way, output files on the host are owned by your host-user while the user inside the container can be root without implying root access on the host system.

## Execute Experiments

### Short Form

An experiment can be started by the following two commands, which leaves all options at their default except for the experiment name.
The first command runs a script that builds the Docker image that serves as the environment for running the experiments in.
This took 224 seconds for us.
The second command runs a script that starts the execution of the specified experiment using the previously built Docker image.
This took between 3h (exp-27) and 33h (exp-48) for us, depending on the experiment.
In case the image is already available to docker, the second command suffices.

```bash
bash experiments/run/build.sh
bash experiments/run/start.sh <experiment-name>
```

Another way to make the image available to docker is to load an image from a file, if you have one available.
The full process then looks like this.

```bash
docker image load -i image.tar
bash experiments/run/start.sh <experiment-name>
```

Also `experiments/start_range.sh` is a script to start multiple experiments in different tmux windows.
Edit line 25 to your liking to define how each run is parameterized and start the experiments by issuing something like

```bash
bash experiments/start_range.sh <start-experiment-number> <end-experiment-number>
```

Therein the experiment numbers are the numbers after `exp-` in the experiment names.

### Full Specification

The full sets of available arguments to the build and start scripts are defined as follows.

```bash
bash experiments/run/build.sh \
-d <definitions-directory-path> \
-b <data-directory-path> \
-p <code-directory-path> \
```

```bash
bash experiments/run/start.sh \
-n <experiment-name> \
-a <host-definitions-path> \
-o <outputs-directory-path> \
-m <container-memory-limit> \
-w <container-swap-memory-limit> \
-c <container-cpu-limit> \
-s <start-stage-index> \
-e <num-stages-to-run>
```

All arguments except `-n <experiment_name>` are optional. The default values are:
- `-n <experiment_name>`: `$1` (if only the name is given as an unnamed argument)
- `-d <container-definitions-path>`: `$(pwd)/experiments/definitions`
- `-b <data-directory-path>`: `$(pwd)/experiments/data`
- `-p <code-directory-path>`: `$(pwd)/code`
- `-a <cache-directory-path>`: `$(pwd)/experiments/cache`
- `-o <container-outputs-path>`: `$(pwd)/experiments/outputs`
- `-m <container-memory-limit>`: (empty)
- `-w <container-swap-memory-limit`: (empty)
- `-c <container-cpu-limit>`: (empty)
- `-s <start-stage-index>`: `0`
- `-e <num-stages-to-run>`: `999`

Empty values of the resource limits mean that the corresponding resource limiting option in the `docker run` command will be omitted entirely.

`<num-stages-to-run>` stages are run, starting with the stage at `<start-stage-index>`.

## License

This project is licensed under the Apache 2.0 License.
