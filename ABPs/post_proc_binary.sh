#!/bin/sh

function removeEmptyLines()
{
    local -r content="${1}"

    echo -e "${content}" | sed '/^\s*$/d'
}

function repeatString()
{
    local -r string="${1}"
    local -r numberToRepeat="${2}"

    if [[ "${string}" != '' && "${numberToRepeat}" =~ ^[1-9][0-9]*$ ]]
    then
        local -r result="$(printf "%${numberToRepeat}s")"
        echo -e "${result// /${string}}"
    fi
}

function isEmptyString()
{
    local -r string="${1}"

    if [[ "$(trimString "${string}")" = '' ]]
    then
        echo 'true' && return 0
    fi

    echo 'false' && return 1
}

function trimString()
{
    local -r string="${1}"

    sed 's,^[[:blank:]]*,,' <<< "${string}" | sed 's,[[:blank:]]*$,,'
}


function printTable()
{
    local -r delimiter="${1}"
    local -r data="$(removeEmptyLines "${2}")"

    if [[ "${delimiter}" != '' && "$(isEmptyString "${data}")" = 'false' ]]
    then
        local -r numberOfLines="$(wc -l <<< "${data}")"

        if [[ "${numberOfLines}" -gt '0' ]]
        then
            local table=''
            local i=1

            for ((i = 1; i <= "${numberOfLines}"; i = i + 1))
            do
                local line=''
                line="$(sed "${i}q;d" <<< "${data}")"

                local numberOfColumns='0'
                numberOfColumns="$(awk -F "${delimiter}" '{print NF}' <<< "${line}")"

                # Add Line Delimiter

                if [[ "${i}" -eq '1' ]]
                then
                    table="${table}$(printf '%s#+' "$(repeatString '#+' "${numberOfColumns}")")"
                fi

                # Add Header Or Body

                table="${table}\n"

                local j=1

                for ((j = 1; j <= "${numberOfColumns}"; j = j + 1))
                do
                    table="${table}$(printf '#| %s' "$(cut -d "${delimiter}" -f "${j}" <<< "${line}")")"
                done

                table="${table}#|\n"

                # Add Line Delimiter

                if [[ "${i}" -eq '1' ]] || [[ "${numberOfLines}" -gt '1' && "${i}" -eq "${numberOfLines}" ]]
                then
                    table="${table}$(printf '%s#+' "$(repeatString '#+' "${numberOfColumns}")")"
                fi
            done

            if [[ "$(isEmptyString "${table}")" = 'false' ]]
            then
                echo -e "${table}" | column -s '#' -t | awk '/^\+/{gsub(" ", "-", $0)}1'
            fi
        fi
    fi
}

echo "Are you running on Longleaf (y/n)?"
read answer

if [ $answer == "y" ]; then
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/klotsa/ABPs/post_proc"
    submit='sbatch'
else
    hoomd_path="$HOME/hoomd-blue/build"
    script_path="$HOME/klotsa/ABPs/post_proc"
    submit='sh'
fi

current=$( date "+%m_%d_%y" )
this_path="$( pwd )"

mkdir ${current}_txt_files
mkdir ${current}_pic_files
mkdir ${current}_vid_files

outpath="$this_path"/"$current"

echo "What is your bin size?"
read bin_size

echo "What is your step size?"
read time_step

echo "What do you want to analyze?"
read method

echo "Do you want to generate plots? (y/n)"
read plot

if [ $method == "lattice_spacing" ]; then
  echo "Do you want to use parallel processing, namely for lattice spacing (y/n)?"
  read parallel
else
  parallel="n"
fi


for file in $(ls *gsd)
do
    if [ "$parallel" = "y" ]; then
        $submit $script_path/analyze_binary_parallel.sh $hoomd_path $outpath $script_path $file $bin_size $time_step $method $plot
    elif [ "$parallel" = "n" ]; then
        $submit $script_path/analyze_binary.sh $hoomd_path $outpath $script_path $file $bin_size $time_step $method $plot
    else
        echo "did not recognize response to parallel processing"
    fi

done
