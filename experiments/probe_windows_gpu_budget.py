"""Read the current process's Windows GPU budgets without changing reservations."""
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from trainer.windows_gpu_memory import query_budgets

if __name__ == '__main__':
    print(json.dumps(query_budgets(), indent=2))
