import yaml

class DotDict(dict):
    def __init__(self, d):
        super().__init__(d)
        for key, value in d.items():
            if isinstance(value, dict):
                value = DotDict(value)
            setattr(self, key, value)

    def __getattr__(self, name):
        return self.get(name)

    def __setattr__(self, name, value):
        self[name] = value
        super().__setattr__(name, value)

class Config:
    @classmethod
    def load(cls, path="parameters.yaml"):
        with open(path, "r") as f:
            cfg = yaml.safe_load(f) or {}

        dot_cfg = DotDict(cfg)

        for key, value in dot_cfg.items():
            setattr(cls, key, value)

            if isinstance(value, DotDict):
                for sub_key, sub_val in value.items():
                    setattr(cls, sub_key, sub_val)


Config.load()
