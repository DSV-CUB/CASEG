import xlsxwriter

def num2col(number):
    return xlsxwriter.utility.xl_col_to_name(number)

class Setup:
    def __init__(self, path_file):
        self.workbook = xlsxwriter.Workbook(path_file)
        self.worksheets = {}
        self.row = {}
        return

    def add_worksheet(self, name):
        self.worksheets[name] = self.workbook.add_worksheet(name)
        self.row[name] = 0
        return

    def write_line(self, name, data, row=None, formating=None):
        if not row is None:
            if row > self.row[name]:
                max_row = row
            else:
                max_row = self.row[name]
            self.row[name] = row
        else:
            max_row = 0

        for i in range(len(data)):
            if formating is None:
                self.worksheets[name].write(self.row[name], i, str(data[i]))
            else:
                self.worksheets[name].write(self.row[name], i, str(data[i]), formating)

        if max_row > self.row[name]:
            self.row[name] = max_row
        else:
            self.row[name] = self.row[name] + 1
        return

    def write_header(self, name, header, row=None):
        header_format = self.workbook.add_format()
        header_format.set_align("center")
        header_format.set_align("vcenter")
        header_format.set_text_wrap()
        header_format.set_bold()
        self.write_line(name, header, row, header_format)
        return

    def set_row(self, name, row, height):
        self.worksheets[name].set_row(row, height)
        return

    def set_freeze_panes(self, name, row, col):
        self.worksheets[name].freeze_panes(row, col)
        return

    def save(self):
        self.workbook.close()
        return