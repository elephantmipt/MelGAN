#!/bin/bash
mkdir -p data
wget -P data "https://data.keithito.com/data/speech/LJSpeech-1.1.tar.bz2"
tar -xvjf data/LJSpeech-1.1.tar.bz2 -C data
rm -rf data/LJSpeech-1.1.tar.bz2