#!/bin/bash
    
    if [ $# -lt 1 ]; then
            echo "usage:"
            echo "       $(basename $0 .sh) [-o outfilename.mp4]  [glob expression]"
            exit 1
    fi

    # output file
    # ----------- 
    if [ "$1" == "-o" ]; then
        shift
        mp4file="$1"
        shift
    else
        mp4file="movie.mp4"
    fi

    # images
    # ------ 
    if [ $# -lt 1 ]; then
        echo "$0: missing image file glob expression"
        exit 2
    fi

    module purge
    module load ffmpeg/5.0

    ulimit -ss unlimited

    # quality values range from 18 (high quality) to 28 (low quality)

    framerate=3
    quality=25
    moviewth=1024
    moviehgt=768
    # movieres="1080p"
    framesize="${moviewth}x${moviehgt}"

    ffmpeg -loglevel debug -threads 6 -pattern_type glob -r $framerate -i "$@" -y -r $framerate -s "$framesize" -c:v libx264 -pix_fmt yuv420p -preset ultrafast -crf $quality ${mp4file}

    exit 0


