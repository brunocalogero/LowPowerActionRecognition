#!/bin/bash

if [ $# -eq 0 ]
then
        echo "No arguments supplied"
        echo "to run, two input arguments needed:"
        echo "input argument 1: 'worker'/'master'/'build'"
        echo "input argument 2: '<your Master node address>'"
        exit;
fi

if [[ "$1" = "master" ]]
then
        pushd seep-system/examples/acita_demo_2015/tmp ; rm masterloger.txt ; java -classpath "../lib/*" uk.ac.imperial.lsds.seep.Main Master `pwd`/../dist/acita_demo_2015.jar Base 2>&1 | tee masterlog.txt ; popd
elif [[ "$1" = "worker" ]]
then
        pushd seep-system/examples/acita_demo_2015/tmp ; rm workerloger.txt ; java -classpath "../lib/*" uk.ac.imperial.lsds.seep.Main Worker 2>&1 | tee workerlog.txt ; popd
else
        echo "Changing MainAddr to local network Master Address"
        pushd seep-system/src/main/resources; python script.py $2 ; popd
        chmod 777 frontier-bld.sh
        ./frontier-bld.sh pi
fi
