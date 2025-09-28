#!/bin/bash

# Default values
IS_CPU=0
IS_CUDA=0
ENV_PATH="gflownet-env"
EXTRAS="all"
DRY_RUN=0

# Allowed extras set
ALL_EXTRAS=("dev" "materials" "molecules" "tree")
VALID_EXTRAS=("${ALL_EXTRAS[@]}" "minimal" "all")

# Display help message and exit if --help is in the arguments
for arg in "$@"; do
    if [[ "$arg" == "--help" ]]; then
		echo "Usage: source install.sh [OPTIONS]"
		echo ""
		echo "Options:"
		echo "  --cpu                 Install CPU-only PyTorch (mutually exclusive with --cuda)."
		echo "  --cuda                Install CUDA-enabled PyTorch (default, and mutually exclusive with --cpu)."
		echo "  --envpath PATH        Path of the Python virtual environment to be installed. Default: ./gflownet-env"
		echo "  --extras LIST         Comma-separated list of extras to install. Default: all. Options:"
		echo "                            - dev: dependencies for development, such as linting and testing packages."
		echo "                            - materials: dependencies for materials applications, such as the Crystal-GFN."
		echo "                            - molecules: dependencies for molecular modelling and generation, such the Conformer-GFN."
		echo "                            - tree: dependencies for training a decision tree sampler."
		echo "                            - all: all of the above"
		echo "                            - minimal: none of the above, that is the minimal set of dependencies."
		echo "  --dry-run             Print the summary of the configuration selected and exit."
		echo "  --help                Show this help message and exit."
		echo ""
		echo "Example:"
		echo "    source install.sh --cpu --envpath ~/myenvs/gflownet-env --extras dev,materials"
		return 0
    fi
done

# Function to check if a value is valid
is_valid_extra() {
    local value="$1"
    for valid in "${VALID_EXTRAS[@]}"; do
        if [[ "$value" == "$valid" ]]; then
            return 0  # Value is valid
        fi
    done
    return 1  # Value is invalid
}

# Validate extras
validate_extras() {
    IFS=',' read -ra EXTRA_VALUES <<< "$1"  # Split input string by comma
    for extra in "${EXTRA_VALUES[@]}"; do
        if ! is_valid_extra "$extra"; then
            echo "Error: Found an invalid value in --extras: '$extra'. Allowed values: ${VALID_EXTRAS[*]}" >&2
            return 1
        fi
    done
}

# Function to get absolute path
get_absolute_path() {
    if [[ "$1" = /* ]]; then
        echo "$1"
    else
        echo "$(pwd)/$1"
    fi
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=1
            ;;
        --cpu)
            IS_CPU=1
            ;;
        --cuda)
            IS_CUDA=1
            ;;
        --envpath)
            shift
            if [[ -z "$1" ]]; then
                echo "Error: --envpath requires a value" >&2
                return 1
            fi
            ENV_PATH=$(get_absolute_path "$1")
            ;;
        --extras)
            shift
            if [[ -z "$1" ]]; then
                echo "Error: --extras requires at least one value" >&2
                return 1
            fi
            EXTRAS="$1"
            ;;
        *)
            echo "Unknown argument: $1" >&2
            return 1
            ;;
    esac
    shift
done

# Ensure mutually exclusive flags --cpu and --cuda
if [[ "$IS_CPU" -eq 1 && "$IS_CUDA" -eq 1 ]]; then
    echo "Error: --cpu and --cuda are mutually exclusive. Please use only one of these flags." >&2
    return 1
fi

# If neither CPU or CUDA are selected, then resort to CUDA by default
if [[ "$IS_CPU" -eq 0 && "$IS_CUDA" -eq 0 ]]; then
    IS_CUDA=1
fi

# Determine installation type
if [[ "$IS_CPU" -eq 1 ]]; then
    INSTALL_TYPE="CPU-only"
fi
if [[ "$IS_CUDA" -eq 1 ]]; then
    INSTALL_TYPE="CUDA-enabled"
fi

# Check if extras are valid
if ! validate_extras $EXTRAS; then
    return 1
fi

# Check if minimal is used alongside other extras
if echo "$EXTRAS" | grep -qE '(^|,)minimal(,|$)' && [[ "$EXTRAS" == *,* ]]; then
    echo "Error: the values in --extras are ambiguous. The extras minimal should be used on its own."
	return 1
fi

# If all is in extras, then set EXTRAS to ALL_EXTRAS
if echo "$EXTRAS" | grep -qE '(^|,)all(,|$)'; then
    IFS=','
    EXTRAS="${ALL_EXTRAS[*]}"
    IFS=' '
fi

# Check if the environment directory exists
if [[ -d "$ENV_PATH" ]]; then
    echo "The environment path '$ENV_PATH' already exists."
    echo "Do you want to continue by updating the existing environment? [Y/n]"
    read -r response

    if [[ "$response" == "Y" ]]; then
        echo "Installation will continue on the existing path " "$ENV_PATH"
    else
        echo "Exiting"
        return 1
    fi
fi

# Print settings selected for the installation
echo -e "The installation will proceed with the following settings:"
echo -e "\t- Path of the Python environment: $ENV_PATH"
echo -e "\t- PyTorch installation will be $INSTALL_TYPE"
echo -e "\t- Sets of dependencies: $EXTRAS"
echo ""

if [[ "$DRY_RUN" -eq 1 ]]; then
    echo "Installation will not proceed because --dry-run was included in the arguments."
	return 1
fi

# Install Python environment
echo "Setting up environment in " "$ENV_PATH"
python -m venv "$ENV_PATH"

# Activate environment
echo "Activating environment"
source "$ENV_PATH""/bin/activate"

# Upgrade pip
echo "Upgrading pip"
python -m pip install --upgrade pip

# Install torch
echo "Installing " "$INSTALL_TYPE" " PyTorch"
if [[ "$IS_CPU" -eq 1 ]]; then
	python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cpu.html
else
	python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
    python -m pip install torch-scatter -f https://data.pyg.org/whl/torch-2.5.1+cu118.html
fi

# Install rest of dependencies
echo "Installing the rest of the dependencies, with extras " "$EXTRAS"
python -m pip install .["$EXTRAS"]

# Exit
echo "Installation of gflownet package completed!"
