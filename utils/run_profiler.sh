#!/bin/bash

# Predefined Nsight Compute metric sets
MEM_METRICS=(
  "dram__bytes.sum"
  "dram__throughput.avg.pct_of_peak_sustained_elapsed"
  "l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"
  "l2__throughput.avg.pct_of_peak_sustained_elapsed"
)

COMPUTE_METRICS=(
  "sm__cycles_active.avg.pct_of_peak_sustained_elapsed"
  "sm__inst_executed.avg.per_cycle_active"
  "smsp__inst_executed.avg.per_cycle_active"
  "sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"
)

# Metric Descriptions Dictionary
declare -A METRIC_DESCRIPTIONS=(
  ["dram__bytes.sum"]="# of bytes accessed in DRAM"
  ["dram__throughput.avg.pct_of_peak_sustained_elapsed"]="DRAM throughput as a percentage of peak sustained elapsed cycles"
  ["l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum"]="# of L1 global load sectors"
  ["l2__throughput.avg.pct_of_peak_sustained_elapsed"]="L2 cache throughput as a percentage of peak sustained elapsed cycles"
  ["sm__cycles_active.avg.pct_of_peak_sustained_elapsed"]="SM active cycles as a percentage of peak sustained elapsed cycles"
  ["sm__inst_executed.avg.per_cycle_active"]="Average number of instructions executed per SM active cycle"
  ["smsp__inst_executed.avg.per_cycle_active"]="Average number of instructions executed per SMSP active cycle"
  ["sm__pipe_fma_cycles_active.avg.pct_of_peak_sustained_elapsed"]="FMA pipe active cycles as a percentage of peak sustained elapsed cycles"
)

get_metrics_for_argset() {
  local argsets=$1
  local combined_metrics=()

  IFS=',' read -ra ADDR <<< "$argsets"
  for set in "${ADDR[@]}"; do
    if [[ "$set" == "memory" ]]; then
      combined_metrics+=("${MEM_METRICS[@]}")
    elif [[ "$set" == "compute" ]]; then
      combined_metrics+=("${COMPUTE_METRICS[@]}")
    else
      echo "Warning: Unknown argset '$set'. Expected 'memory' or 'compute'." >&2
    fi
  done

  # Join metrics with comma
  local IFS=","
  echo "${combined_metrics[*]}"
}

NCU_ARGS=()
BIN_ARGS=("--skip_correctness_tests")
ARGSET_VAL="memory,compute" # default
METRICS_STR=""

while [[ $# -gt 0 ]]; do
  case $1 in
    -h|--help)
      echo "Description: Runs the compiled test binary under NVIDIA Nsight Compute with predefined"
      echo "             metric sets to help you analyze the performance and bottlenecks of your kernel."
      echo "             Correctness tests are skipped when profiling."
      echo ""
      echo "Usage: $0 [options]"
      echo ""
      echo "Profiler Options:"
      echo "  --ncu_arg <arg>         Pass raw argument to Nsight Compute"
      echo "  --ncu_argset <sets>     Comma-separated list of metric sets (memory, compute) [default: memory,compute]"
      echo "  -h, --help              Show this help message"
      echo ""
      echo "Main Driver Options:"
      "$(dirname "$0")/run_test" -h | grep -v '^Usage:' | grep -v '^Options:' | grep -v '\-h, --help' | grep -v 'skip_correctness_tests'
      exit 0
      ;;
    --ncu_arg)
      if [[ "$2" == "--metrics" && -n "$3" ]]; then
        # Capture raw metrics provided directly to ncu via --ncu_arg
        if [[ -n "$METRICS_STR" ]]; then
          METRICS_STR="${METRICS_STR},$3"
        else
          METRICS_STR="$3"
        fi
      fi
      NCU_ARGS+=("$2")
      shift 2
      ;;
    --ncu_argset=*)
      ARGSET_VAL="${1#*=}"
      shift
      ;;
    --ncu_argset)
      ARGSET_VAL="$2"
      shift 2
      ;;
    *)
      BIN_ARGS+=("$1")
      shift
      ;;
  esac
done

ARGSET_METRICS=$(get_metrics_for_argset "$ARGSET_VAL")

if [[ -n "$ARGSET_METRICS" ]]; then
  NCU_ARGS=("--metrics" "$ARGSET_METRICS" "${NCU_ARGS[@]}")
  if [[ -n "$METRICS_STR" ]]; then
    METRICS_STR="${ARGSET_METRICS},${METRICS_STR}"
  else
    METRICS_STR="$ARGSET_METRICS"
  fi
fi

# Run Nsight Compute and capture the exit code
ncu "${NCU_ARGS[@]}" "$(dirname "$0")/run_test" "${BIN_ARGS[@]}"
EXIT_CODE=$?

# Print the Metric Guide at the very end
echo ""
echo -e "\033[1;36m================================================================================\033[0m"
echo -e "\033[1;36m                            Nsight Compute Metric Guide                         \033[0m"
echo -e "\033[1;36m================================================================================\033[0m"

# We split METRICS_STR by comma and deduplicate to avoid printing descriptions twice
IFS=',' read -ra ADDR <<< "$METRICS_STR"
declare -A PRINTED
for metric in "${ADDR[@]}"; do
  if [[ -z "${PRINTED[$metric]}" ]]; then
    PRINTED[$metric]=1
    desc="${METRIC_DESCRIPTIONS[$metric]}"
    if [[ -n "$desc" ]]; then
      printf "\033[1m%-75s\033[0m\n  %s\n\n" "$metric" "$desc"
    fi
  fi
done

exit $EXIT_CODE
