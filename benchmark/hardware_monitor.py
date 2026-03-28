from jtop import jtop
import time
import threading
import psutil
from config import TARGET_DEVICE
class HardwareMonitor:
    '''
    A class to monitor hardware metrics such as GPU utilization, power consumption, temperature, and RAM usage on NVIDIA Jetson devices using the jtop library.
    The monitor runs in a separate thread to continuously collect data while the main inference pipeline is running
    
    Attributes:
        interval (float): The time interval (in seconds) between each hardware metric collection.
        running (bool): A flag to control the monitoring thread.
        gpu_util (list): A list to store GPU utilization percentages.
        power (list): A list to store power consumption values.
        temp (list): A list to store temperature values.
        ram (list): A list to store RAM usage values.
    methods:
        start(): Starts the hardware monitoring thread.
        stop(): Stops the hardware monitoring thread and waits for it to finish.
        summary(): Returns a dictionary containing summary statistics of the collected hardware metrics, such as average and peak GPU utilization, power consumption, temperature, and RAM usage.

    Example usage:
        monitor = HardwareMonitor(interval=0.1)
        monitor.start()
        # Run your inference pipeline here
        monitor.stop()
        stats = monitor.summary()
        print(stats)
    '''

    def __init__(self, interval=0.1):
        self.interval = interval
        self.running = False

        self.gpu_util = []
        self.power = []
        self.temp = []
        self.ram = []

    def _monitor(self):
        '''The internal method that runs in a separate thread to continuously collect hardware metrics using jtop. It collects GPU utilization, power consumption, temperature, and RAM usage at the specified interval until the monitoring is stopped.
        '''
        with jtop() as jetson:
            while self.running and jetson.ok():

                stats = jetson.stats

                self.gpu_util.append(stats.get("GPU", 0))
                self.ram.append(get_ram())

                temps = jetson.temperature
                if TARGET_DEVICE == "cuda":
                    if "gpu" in temps:
                        self.temp.append(temps["gpu"]["temp"])
                else:
                    if "cpu" in temps:
                        self.temp.append(temps["cpu"]["temp"])

                p = jetson.power
                if "tot" in p:
                    self.power.append(p["tot"]["power"])

                time.sleep(self.interval)

    def start(self):
        '''
         Starts the hardware monitoring thread by setting the running flag to True and creating a new thread that targets the _monitor method.
        '''

        self.running = True
        self.thread = threading.Thread(target=self._monitor)
        self.thread.start()

    def stop(self):
        ''' Stops the hardware monitoring thread by setting the running flag to False and waiting for the thread to finish using join().
        '''
        self.running = False
        self.thread.join()

    def summary(self) -> dict:
        ''' Returns a dictionary containing summary statistics of the collected hardware metrics, such as average and peak GPU utilization, power consumption, temperature, and RAM usage. It uses numpy to calculate the mean and max values for each metric.

        Returns:
            dict: A dictionary containing summary statistics of the collected hardware metrics, including average and peak GPU utilization, power consumption, temperature, and RAM usage.
                - gpu_util_avg: Average GPU utilization percentage.
                - gpu_util_peak: Peak GPU utilization percentage.
                - power_avg: Average power consumption in watts.
                - power_peak: Peak power consumption in watts.
                - temp_peak: Peak temperature in degrees Celsius.
                - ram_peak: Peak RAM usage in megabytes.
        '''
        import numpy as np
        if not self.ram:
            return {}

        
        return {
            "gpu_util_avg": np.mean(self.gpu_util),
            "gpu_util_peak": np.max(self.gpu_util),

            "power_avg": np.mean(self.power),
            "power_peak": np.max(self.power),

            "temp_peak": np.max(self.temp),

            "ram_peak": np.max(self.ram),
        }


def get_ram():
    return psutil.virtual_memory().used / (1024 * 1024)