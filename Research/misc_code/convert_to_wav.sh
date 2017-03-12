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
    num_flacs=$(ls $recordings | grep -Ev '.wav' | wc -l | awk '{ print $1 }')
    if [ $num_flacs -gt 0 ]; then # check if there's at least one flac file
        echo -e "Conversion from flac to wav error log\nRan on $(date +"%d-%m-%y") at $(date +"%T") by $(whoami)" > .convert_to_wav.log
        echo "$Found $num_flacs in $recordings. Converting files to .wav"
        sleep 1
        i=1
        files=$(ls $recordings | grep -vE '\.wav$')
        for f in $files
        do
            echo "$i of $num_flacs"
            #echo $f
            wavname="$(echo "$f" | cut -f 1 -d '.').wav"
            echo -e "converting $recordings/$f to $recordings/$wavname"
            ffmpeg -y -i "$recordings/$f" "$recordings/$wavname" &>/dev/null #conversion and no output

            if [ $? -eq 0 ]; then #check error in Conversion
                echo "CONVERSION SUCCEEDED"
            else
                echo "Failed to convert $file" >> .convert_to_wav.log #log errors
            fi
            ((i++))
        done
        
        errcheck=$(wc -l < .convert_to_wav.log | awk '{ print $1 }') 
        if [ $errcheck -eq 2 ]; then #check for errors in log file
            echo -e "\nConversion of all files completed correctly"
            echo -e "No errors found.\n\nDeleting .convert_to_wav.log"
            rm .convert_to_wav.log
        else
            echo "Some files were not converted successfully. Check .convert_to_wav.log for details."
        fi
    else
        echo "No .flac files to convert in recordings folder"
    fi
    
    while true; do
        read -p "Would you like to delete all non-wav duplicates of recordings?(y/n): " ans

        if [ "$ans" == 'y' ] || [ "$ans" == 'Y' ]; then
            echo -e "All duplicate recordings will be deleted. Deleting in 2 seconds..."
            sleep 2
            filesDelete=$(ls $recordings | grep -Ev '\.wav$')
            for f in $filesDelete; do
                echo "removing $recordings/$f"
                rm "$recordings/$f"
            done
            numFiles=$(ls $recordings | wc -l | awk '{ print $1}')
            if [ $numFiles -eq $i ]; then # need to fix error since i will be +1 
                echo "All files were deleted successfully. Exiting..."
                exit 0
            else
                echo "Some files in $recordings could not be deleted."
                exit 1
            fi
            break
        elif [ $ans == 'N' ] || [ "$REPLY" == 'n' ]; then 
            numFiles=$(ls $recordings | wc -l | awk '{ print $1}')
            num=$(($numFiles - $i))
            echo "There are $num duplicates. Exiting..."
            break
            exit 0
        else
            echo "invalid option"
            clear
            echo "Would you like to delete all non-wav duplicates of recordings?: "
        fi
    done
    
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
