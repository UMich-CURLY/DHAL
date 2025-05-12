import copy
import pickle as pkl

import numpy as np
import torch


def class_to_dict(obj) -> dict:
    if not hasattr(obj, "__dict__"):
        return obj
    result = {}
    for key in dir(obj):
        if key.startswith("_") or key == "terrain":
            continue
        element = []
        val = getattr(obj, key)
        if isinstance(val, list):
            for item in val:
                element.append(class_to_dict(item))
        else:
            print(key)
            element = class_to_dict(val)
        result[key] = element
    return result


class MultiLogger:
    def __init__(self):
        self.loggers = EpisodeLogger()
        
    def log(self, info):
        self.loggers.log(info)

    def save(self, filename):
        with open(filename, 'wb') as file:
            logdict = self.loggers.infos
            pkl.dump(logdict, file)
            print(f"\nSaved log!; Path: {filename}")

    def reset(self):
        for key, log in self.loggers.items():
            log.reset()


class EpisodeLogger:
    def __init__(self):
        self.infos = []

    def log(self, info):
        for key in info.keys():
            if isinstance(info[key], torch.Tensor):
                info[key] = info[key].detach().cpu().numpy()

            if isinstance(info[key], dict):
                continue
            elif "image" not in key:
                info[key] = copy.deepcopy(info[key])

        self.infos += [dict(info)]

    def reset(self):
        self.infos = []
