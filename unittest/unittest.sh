#!/bin/bash
# Copyright (c) 2022 Innodisk Crop.
# 
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT


PRINTUSAGE() {
    echo "usage: ./unittest.sh [TEST]"
    echo "[TEST]"
    echo "        dpu:  run the dpu unit test"
    echo "        util: run the utility unit test"
    echo "        yolo: run the yolo unit test"
    echo "        cnn:  run the cnn unit test"
    echo "        all:  run all off the dpu unit test"
}

if [ ! -z $1 ]; then
    case $1 in
        dpu)
            FILE="dpuTest.py"
            ;;
        util)
            FILE="utilTest.py"
            ;;
        yolo)
            FILE="yoloTest.py"
            ;;
        cnn)
            FILE="cnnTest.py"
            ;;
        all)
            TF_CPP_MIN_LOG_LEVEL=3 python3 -m unittest discover . "*Test.py" -v
            exit 0
            ;;
        *)
            echo 1>&2 "Unsupported argument: $1"
            PRINTUSAGE
            exit 1
            ;;
    esac

    TF_CPP_MIN_LOG_LEVEL=3 python3 -m unittest ${FILE} -v

else
    PRINTUSAGE
fi