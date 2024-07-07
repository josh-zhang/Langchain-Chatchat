import time
import subprocess
import threading
import concurrent.futures

from configs import logger


class PythonScriptExecutor:
    def __init__(self):
        # Constructor for any initialization if needed
        pass

    def execute_script(self, script_path):
        """
        Executes a Python script and captures its output, error, and execution time.
        """
        start_time = time.time()

        # Execute the script using subprocess
        with subprocess.Popen(f"python3.10 {script_path}", stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              shell=True) as process:
            for line in process.stdout:
                logger.info(line.strip())

        end_time = time.time()
        duration = end_time - start_time

        # Logging the outcome
        if process.returncode == 0:
            logger.info(f"Script {script_path} executed successfully in {duration:.2f} seconds.")
        else:
            logger.error(f"Script {script_path} failed")

        # Returning the results in a structured format
        return {
            'return_code': process.returncode,
            'execution_time': duration
        }


JobExecutor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
FuturesAtomic = threading.Lock()
JobFutures = {}
