# Smoothness vs Gradient norm

A pytorch implementation for the LSTM experiments in the paper: Why Gradient Clipping Accelerates Training: A Theoretical Justification for Adaptivity

This repo is based on the repo at https://github.com/salesforce/awd-lstm-lm. Please refer to the original repo for a detailed setup description.

For data setup, run `./getdata.sh`.

To reproduce the result of estimating smoothness vs gradient norm on AWD-LSTM training with PTB, simply run `CUDA_VISIBLE_DEVICES=1 python main.py --epochs 2`

The smoothness and gradient norm data collected along training is stored as csv files in side the ./ckpts folder.
