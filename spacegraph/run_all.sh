
#!/bin/bash
expname="experiments_1028"

mkdir -p ${expname}/run_logs

for script in ${expname}/scripts/*.sh; do
    name=$(basename "$script" .sh)

    echo "===== Starting $name ====="

    bash "$script" > "${expname}/run_logs/${name}.log" 2>&1

    status=$?

    if [ "$status" -eq 0 ]; then
        echo "SUCCESS: $name"
    else
        echo "FAILED: $name (exit code $status)"
    fi
done

echo "All scripts executed. Check ${expname}/run_logs for details."
