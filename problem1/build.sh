#!/bin/bash

jupytext --to notebook pipeline.py
jupyter nbconvert --to html --execute pipeline
