import json
import pathlib
from dataclasses import dataclass

class HelperObject:
    """Helper class to convert json into Python object """
    def __init__(self,dct):
        self.__dict__.update(dct)


class SingletonMeta(type):
    """
    The Singleton class can be implemented in different ways in Python. Some
    possible methods include: base class, decorator, metaclass. We will use the
    metaclass because it is best suited for this purpose.
    """

    _instances = {}

    def __call__(cls, *args, **kwargs):
        """
        Possible changes to the value of the `__init__` argument do not affect
        the returned instance.
        """
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]

@dataclass
class Paths(metaclass=SingletonMeta):
    """converts paths to pathlib paths"""
   
    def __init__(self,paths):
        paths_dict = vars(paths)
        for key in paths_dict:
            setattr(self,key, self.convert_to_pathlib(paths_dict[key]))

    def convert_to_pathlib(self,path):
        return pathlib.Path(path)

class Config:
    """config class with contains data, train and model hyperparameters"""


    def __init__(self,data,train,model,paths):
        self.data = data
        self.train = train
        self.model = model
        self.paths = paths
    
    @classmethod
    def from_json(cls,cfg):
        """creates config from json"""
        params = json.loads(json.dumps(cfg),object_hook=HelperObject)
        # add to params path pathlib style 
        params.paths = Paths(params.paths)
        return cls(params.data,params.train,params.model,params.paths)
    