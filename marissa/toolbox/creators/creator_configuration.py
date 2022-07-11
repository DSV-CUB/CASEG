import pickle
import os
from marissa.toolbox.tools import tool_general


class Inheritance:
    def __init__(self):
        self.path_rw = os.path.dirname(os.path.realpath(__file__)).replace("/", "\\").replace("marissa\\marissa\\toolbox\\creators", "marissa\\appdata")
        self.name = ""
        self.examination = None  # human, phantom
        self.tissue = None  # myocardium, ...
        self.view = None  # SAXMV, 3CV, ...
        self.measure = None  # T1 MAP, ...
        return

    def set(self, **kwargs):
        for key, value in kwargs.items():
            if hasattr(self, key) and not tool_general.check_class_has_method(self, key) and not key == "path":
                exec("self." + key + " = kwargs.get(\"" + key + "\", None)")
        return True

    def save(self, path=None, timestamp=""):
        if path is None:
            save_to = self.path_rw
        else:
            save_to = path

        if self.examination is None or self.tissue is None or self.view is None or self.measure is None:
            raise ValueError("In the configuration examination, tissue, view and measure cannot be None, but at least one of them is None.")
        elif save_to.endswith(".pickle"):
            with open(save_to, 'wb') as file:
                pickle.dump(self, file)
                file.close()
        else:
            directory = self.examination.upper() + "_" + self.measure.upper() + "_" + self.tissue.upper() + "_" + self.view.upper() + ("_" + timestamp if not timestamp == "" else "")

            if not save_to.endswith(directory):
                save_to = save_to + "\\" + directory

            os.makedirs(save_to, exist_ok=True)

            self.set(path_rw=save_to.replace(directory, "")[:-1])

            with open(save_to + "\\" + self.name + ".pickle", 'wb') as file:
                pickle.dump(self, file)
                file.close()

        return True

    def load(self, path=None):
        if path is None:
            load_from = self.path_rw + "\\" + self.examination.upper() + "_" + self.measure.upper() + "_" + self.tissue.upper() + "_" + self.view.upper() + "\\" + self.name + ".pickle"
        else:
            load_from = path

        with open(load_from, 'rb') as file:
            obj = pickle.load(file)
            file.close()

        if type(self) == type(obj):
            for key in obj.__dict__:
                if key in self.__dict__ and key not in ["version", "author", "contact", "date"]:
                    self.__dict__[key] = obj.__dict__[key]
            self.path_rw = load_from
        else:
            raise TypeError("The loaded object is not a configuration object")

        #self.__dict__.clear()
        #self.__dict__.update(obj.__dict__)

        return True

    def reset(self):
        self.__init__()
        return

