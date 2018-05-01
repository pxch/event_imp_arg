from copy import deepcopy

from lxml import etree

from rich_tree_pointer import RichTreePointer
from utils import check_type


class ImpArgInstance(object):
    def __init__(self, pred_pointer, arguments):
        check_type(pred_pointer, RichTreePointer)
        self.pred_pointer = pred_pointer
        self.arguments = arguments

    def __eq__(self, other):
        return str(self.pred_pointer) == str(other.pred_pointer)

    def __ne__(self, other):
        return not self.__eq__(other)

    @classmethod
    def parse(cls, text):
        root = etree.fromstring(text)
        pred_pointer = RichTreePointer.from_text(root.get('for_node'))
        arguments = []
        for child in root:
            label = child.get('value')
            arg_pointer = RichTreePointer.from_text(child.get('node'))
            attribute = ''
            if len(child[0]) > 0:
                attribute = child[0][0].text

            arguments.append((label, deepcopy(arg_pointer), attribute))
        return cls(pred_pointer, arguments)

    def __str__(self):
        xml_string = '<annotations for_node="{}">'.format(self.pred_pointer)
        for label, arg_pointer, attribute in self.arguments:
            xml_string += \
                '<annotation value="{}" node="{}">'.format(label, arg_pointer)
            xml_string += '<attributes>{}</attributes>'.format(
                '<attribute>{}</attribute>'.format(attribute)
                if attribute else '')
            xml_string += '</annotation>'
        xml_string += '</annotations>'
        return xml_string
