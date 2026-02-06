import re
import sys
import copy
import yaml

class Config(dict):

    def __init__(self, *args, **kw):

        super(Config,self).__init__(*args, **kw)

    def overlay(self, hash, root=None):

        if root is None: root = self

        for key in hash:

            if key not in root:
                if isinstance(hash[key], dict):
                    root[key] = copy.deepcopy(hash[key])
                else:
                    root[key] = hash[key]
            elif isinstance(hash[key],dict) and isinstance(root[key],dict):
                self.overlay(hash[key], root[key])
            else:
                root[key] = hash[key]

    def read(self, file):

        with open(file, 'r') as ymlfile:
            hash = yaml.load(ymlfile,Loader=yaml.SafeLoader)

        self.overlay(hash)

        return hash
