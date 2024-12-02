#!/bin/bash

# Function to display usage
usage() {
    echo "Usage: $0 -f <python_file> -e <python_environment> \n
    -f: Python file to run \n
    -e: Optional: Python environment to source before running the file. Is the full path to Activate file of the environment. \n"
    exit 1
}

# Parse command line arguments
while getopts ":f:e:" opt; do
    case ${opt} in
        f)
            PYTHON_FILE=$OPTARG
            ;;
        e )
            PYTHON_ENVIRONMENT=$OPTARG
            ;;
        \? )
            usage
            ;;
    esac
done

# Check if both variables are set
if [ -z "$PYTHON_FILE" ]; then
    usage
fi

# Export the python file
export PYTHON_FILE

# Export the environment if it's provided, otherwise skip it
if [ -n "$PYTHON_ENVIRONMENT" ]; then
    export PYTHON_ENVIRONMENT
    # Submit the job with environment
    bsub < run.sh -env "PYTHON_FILE=$PYTHON_FILE,PYTHON_ENVIRONMENT=$PYTHON_ENVIRONMENT"
else
    # Submit the job without environment
    bsub < run.sh -env "PYTHON_FILE=$PYTHON_FILE"
fi