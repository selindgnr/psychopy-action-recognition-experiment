# PsychoPy Action Recognition Experiment

This repository contains a PsychoPy-based human action recognition experiment. Participants watch short video clips and classify the observed action using number keys `1-5`.

Suggested repository name: `psychopy-action-recognition-experiment`

## Experiment Overview

- Platform: PsychoPy Builder / PsychoPy Python export
- Task: Video-based action recognition
- Action classes: `JumpingJack`, `Lunges`, `PullUps`, `PushUps`, `Swing`
- Flow: instructions -> training phase -> test phase -> performance summary

## Main Files

- `action_recognition_experiment.psyexp`: main PsychoPy Builder file
- `action_recognition_experiment.py`: exported PsychoPy experiment script
- `ucf5_training_conditions.csv`: training-phase condition file
- `sorted_test_list.csv`: ordered test-phase stimulus list
- `videos.xlsx`: video inventory

## Helper Scripts

- `master_test_list.py`: builds a video inventory from the stimulus folders and writes `videos.xlsx`
- `final_test_list_ordered.py`: creates a balanced test list and writes `sorted_test_list.csv`
- `concise_csv_extractor.py`: converts large PsychoPy output files into a shorter analysis-friendly format

## Folder Structure

- `training/`: training videos
- `ucf5/`: main test video pool

## GitHub Notes

This repository is configured with `.gitignore` to keep large stimulus videos and generated runtime outputs out of version control. That keeps the GitHub repository smaller and easier to manage.

Recommended to keep in Git:

- `*.psyexp`
- main Python scripts
- condition CSV files
- analysis / helper scripts
- this README

Recommended to keep out of Git:

- raw stimulus videos
- participant output files
- auto-generated files such as `*_lastrun.py`
- temporary or backup working folders

If you want to store stimulus videos on GitHub as well, Git LFS is a better option than regular Git.

## Running the Experiment

1. Install PsychoPy.
2. Make sure the required stimulus videos are present under `training/` and `ucf5/`.
3. Run `action_recognition_experiment.psyexp` in PsychoPy Builder, or run `action_recognition_experiment.py` with Python.

## Status

The repository has been cleaned for GitHub upload and now keeps the core experiment files, condition files, and helper scripts while excluding large generated or backup content.
