from dataclasses import dataclass

@dataclass
class Configuration:
    fw_get_tensor: int
    fw_set_tensor: int

def getconfig():
    return Configuration(
        fw_set_tensor=5,
        fw_get_tensor=3
    )