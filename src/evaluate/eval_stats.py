import abc
from collections import OrderedDict

from texttable import Texttable


def get_table(header, content):

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_cols_align(['c', 'c', 'c', 'c', 'c'])
    table.set_cols_valign(['m', 'm', 'm', 'm', 'm'])
    table.set_cols_width([15, 10, 10, 15, 15])
    table.set_precision(2)

    table.header(header)
    for row in content:
        table.add_row(row)

    return table


class AccuracyStats(object):
    def __init__(self):
        self.num_cases = 0
        self.num_positives = 0
        self.num_choices = []

    def reset(self):
        self.num_cases = 0
        self.num_positives = 0
        self.num_choices = []

    def add_eval_result(self, correct, num_choices):
        self.num_cases += 1
        self.num_positives += correct
        self.num_choices.append(num_choices)
        # self.check_consistency()

    def check_consistency(self):
        assert len(self.num_choices) == self.num_cases, \
            'Length of num_choices not equal to number of cases.'
        assert 0 <= self.num_positives <= self.num_cases, \
            'Number of positive cases must be positive and ' \
            'smaller than number of cases'

    def add_accuracy_stats(self, accuracy_stats):
        self.num_cases += accuracy_stats.num_cases
        self.num_positives += accuracy_stats.num_positives
        self.num_choices.extend(accuracy_stats.num_choices)
        # self.check_consistency()

    def get_accuracy(self):
        if self.num_cases != 0:
            return float(self.num_positives) * 100. / self.num_cases
        else:
            return 0.0

    def get_avg_choices(self):
        if self.num_cases != 0:
            return float(sum(self.num_choices)) / self.num_cases
        else:
            return 0.0

    def __str__(self):
        return '{}/{}'.format(self.num_positives, self.num_cases)

    def pretty_print(self):
        result = '{} correct in {} .'.format(
            self.num_positives, self.num_cases)
        result += ' Accuracy = {:.2f}% .'.format(self.get_accuracy())
        result += ' Avg # of choices = {:.2f} .'.format(self.get_avg_choices())
        return result


class AccuracyStatsGroup(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self, desc, key_list):
        self.desc = desc
        self.accuracy_dict = OrderedDict()
        for key in key_list:
            self.accuracy_dict[key] = AccuracyStats()

    def add_eval_result(self, key, correct, num_choices):
        if key not in self.accuracy_dict:
            self.accuracy_dict[key] = AccuracyStats()
        self.accuracy_dict[key].add_eval_result(correct, num_choices)

    def reset(self):
        for accuracy in self.accuracy_dict.values():
            accuracy.reset()

    def __str__(self):
        result = 'Accuracy by {}\n'.format(self.desc)
        for key, accuracy in self.accuracy_dict.items():
            result += '{}: {}\n'.format(key, str(accuracy))
        return result

    def pretty_print(self):
        result = 'Accuracy by {}\n'.format(self.desc)
        for key, accuracy in self.accuracy_dict.items():
            result += '{}: {}\n'.format(key, accuracy.pretty_print())
        return result

    def print_table(self):
        header = [self.desc, '# Cases', '# Correct', 'Accuracy (%)',
                  'Avg # Choices']

        content = []
        for key, accuracy in self.accuracy_dict.items():
            row = [
                key,
                accuracy.num_cases,
                accuracy.num_positives,
                accuracy.get_accuracy(),
                accuracy.get_avg_choices()
            ]
            content.append(row)

        table = get_table(header, content)
        print table.draw()


class EvalStats(object):
    def __init__(self):
        self.accuracy_all = AccuracyStats()
        self.accuracy_group_dict = OrderedDict()

    def add_accuracy_group(self, name, accuracy_group):
        assert name not in self.accuracy_group_dict
        self.accuracy_group_dict[name] = accuracy_group

    def add_eval_result(self, correct, num_choices, **kwargs):
        self.accuracy_all.add_eval_result(correct, num_choices)
        if kwargs is not None:
            for name, key in kwargs.iteritems():
                if name in self.accuracy_group_dict:
                    self.accuracy_group_dict[name].add_eval_result(
                        key, correct, num_choices)

    def reset(self):
        self.accuracy_all.reset()
        for accuracy_group in self.accuracy_group_dict.values():
            accuracy_group.reset()

    def __str__(self):
        result = 'All: {}\n'.format(self.accuracy_all)
        for accuracy_group in self.accuracy_group_dict.values():
            result += str(accuracy_group)
        return result

    def pretty_print(self):
        result = 'All: {}\n'.format(self.accuracy_all.pretty_print())
        for accuracy_group in self.accuracy_group_dict.values():
            result += accuracy_group.pretty_print()
        return result

    def print_table(self):
        header = ['', '# Cases', '# Correct', 'Accuracy (%)', 'Avg # Choices']

        content = [[
            'All',
            self.accuracy_all.num_cases,
            self.accuracy_all.num_positives,
            self.accuracy_all.get_accuracy(),
            self.accuracy_all.get_avg_choices()
        ]]

        table = get_table(header, content)
        print
        print table.draw()

        for accuracy_group in self.accuracy_group_dict.values():
            accuracy_group.print_table()
