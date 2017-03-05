#! /bin/bash

#print script usage
usage()
{
    echo "Bash script to convert .flac files in a directory to .wav files"
    echo "Usage:"
    echo -e "-p or --path: path to the directory where the .flac files are converted.\nThis path can be relative(e.g. ../<your_path>)."
    echo "-h or --help: Help information and usage"
    exit 0
}

#convert all flac files to wav files
convert()
{
    recordings=$1 # path from argument
    num_flacs=$(ls $recordings | grep .flac | wc -l | awk '{ print $1 }')
    if [ $num_flacs -gt 0 ]; then # check if there's at least one flac file
        echo -e "Conversion from flac to wav error log\nRan on $(date +"%d-%m-%y") at $(date +"%T") by $(whoami)" > .flac_convert_errors.log
        echo "$Found $num_flacs in $recordings. Converting files to .wav"
        sleep 1
        i=1
        for f in $(ls $recordings | grep .flac)
        do
            echo "$i of $num_flacs"
            wavname="$(echo $f | cut -f 1 -d '.').wav"
            echo -e "converting $recordings/$f to $recordings/$wavname"
            ffmpeg -y -i "$recordings/$f" "$recordings/$wavname" &>/dev/null #conversion and no output

            if [ $? -eq 0 ]; then #check error in Conversion
                echo "CONVERSION SUCCEEDED"
            else
                echo "Failed to convert $file" >> .flac_convert_errors.log #log errors
            fi
            ((i++))
        done

        errcheck=$(wc -l < .flac_convert_errors.log | awk '{ print $1 }') 
        if [ $errcheck -eq 2 ]; then #check for errors in log file
            echo -e "\nConversion of all files completed correctly"
            echo -e "No errors found.\n\nDeleting .flac_convert_errors.log"
            rm .flac_convert_errors.log
        else
            echo "Some files were not converted successfully. Check .flac_convert for details."
        fi
    else
        echo "No .flac files to convert in recordings folder"
    fi
}

if test $# -eq 0; then
    echo "Error. This script needs at least one argument. Use flac_convert -h for help."
else
    while test $# -gt 0; do # capture arguments
        case $1 in 
            -h|--help)
                usage
                ;;
            -p|--path)
                opt=$1
                shift
                if test $# -gt 0; then
                    convert $1
                else
                    echo -e "Error! Option $opt needs a value.\nUse $0 -h for usage."
                    exit 1
                fi
                ;;
            *)
                break
                ;;
        esac
    done
fi
