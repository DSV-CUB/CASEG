import os
from marissa.toolbox.tools import tool_general
from marissa.toolbox.creators import creator_configuration


class Setup(creator_configuration.Inheritance):
    def __init__(self, **kwargs):
        super().__init__()
        self.name = "configuration_modules_datahandler_dicom"
        self.examination = None  # human, phantom
        self.tissue = None  # myocardium, ...
        self.view = None  # MVSAX, 3CV, ...
        self.measure = None  # T1 MAP, ...

        self.path = os.path.dirname(os.path.realpath(__file__))
        self.path_toi = os.path.join(self.path, "tags_of_interest.txt")
        self.path_classes = os.path.join(self.path, "classes.txt")

        dcm_toi = tool_general.read_file_and_split(self.path_toi, "# START\n", "# END\n")
        self.toi_addresses = [int(x[1:9], 16) for x in [tag.replace(',', '') for tag in [element[0] for element in dcm_toi]]]
        self.toi_parameter = [x == "x" for x in [element[1] for element in dcm_toi]]
        self.toi_description = [x for x in [element[2] for element in dcm_toi]]

        self.filter = "RELEVANT"

        self.set(**kwargs)
        return
