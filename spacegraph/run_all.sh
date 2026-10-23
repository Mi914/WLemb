
#!/bin/bash

mkdir -p experiments_1022/run_logs

for script in experiments_1022/scripts/*.sh; do
    name=$(basename "$script" .sh)

    echo "===== Starting $name ====="

    bash "$script" > "experiments_1022/run_logs/${name}.log" 2>&1

    status=$?

    if [ "$status" -eq 0 ]; then
        echo "SUCCESS: $name"
    else
        echo "FAILED: $name (exit code $status)"
    fi
done
