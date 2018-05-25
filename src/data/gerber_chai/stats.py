from collections import defaultdict
from operator import itemgetter

from texttable import Texttable

from helper import compute_f1


def print_stats(all_predicates, verbose=0):
    predicates_by_pred = defaultdict(list)

    for predicate in all_predicates:
        predicates_by_pred[predicate.n_pred].append(predicate)

    num_dict = {}

    for n_pred, predicates in predicates_by_pred.items():
        num_dict[n_pred] = [len(predicates)]
        num_dict[n_pred].append(
            sum([predicate.num_imp_arg() for predicate in predicates]))
        if verbose >= 1:
            num_dict[n_pred].append(
                sum([predicate.num_imp_arg(2) for predicate in predicates]))
            num_dict[n_pred].append(
                sum([predicate.num_oracle() for predicate in predicates]))
        if verbose >= 2:
            num_dict[n_pred].append(
                sum([1 for predicate in predicates if
                     'arg0' in predicate.imp_args]))
            num_dict[n_pred].append(
                sum([1 for predicate in predicates if
                     'arg1' in predicate.imp_args]))
            num_dict[n_pred].append(
                sum([1 for predicate in predicates if
                     'arg2' in predicate.imp_args]))
            num_dict[n_pred].append(
                sum([1 for predicate in predicates if
                     'arg3' in predicate.imp_args]))
            num_dict[n_pred].append(
                sum([1 for predicate in predicates if
                     'arg4' in predicate.imp_args]))

    total_pred = 0
    total_arg = 0
    total_arg_in_range = 0
    total_oracle_arg = 0
    total_imp_arg0 = 0
    total_imp_arg1 = 0
    total_imp_arg2 = 0
    total_imp_arg3 = 0
    total_imp_arg4 = 0

    table_content = []

    for n_pred, num in num_dict.items():
        table_row = [n_pred] + num[:2]
        table_row.append(float(num[1]) / num[0])
        if verbose >= 1:
            table_row.append(num[2])
            table_row.append(num[3])
            table_row.append(100. * float(num[3]) / num[1])
        if verbose >= 2:
            table_row += num[4:]
        table_content.append(table_row)

        total_pred += num[0]
        total_arg += num[1]
        if verbose >= 1:
            total_arg_in_range += num[2]
            total_oracle_arg += num[3]
        if verbose >= 2:
            total_imp_arg0 += num[4]
            total_imp_arg1 += num[5]
            total_imp_arg2 += num[6]
            total_imp_arg3 += num[7]
            total_imp_arg4 += num[8]

    table_content.sort(key=itemgetter(2), reverse=True)
    table_row = [
        'Overall',
        total_pred,
        total_arg,
        float(total_arg) / total_pred
    ]
    if verbose >= 1:
        table_row.extend([
            total_arg_in_range,
            total_oracle_arg,
            100. * float(total_oracle_arg) / total_arg
        ])
    if verbose >= 2:
        table_row.extend([
            total_imp_arg0,
            total_imp_arg1,
            total_imp_arg2,
            total_imp_arg3,
            total_imp_arg4
        ])
    table_content.append([''] * len(table_row))
    table_content.append(table_row)

    table_header = ['Pred.', '# Pred.', '# Imp.Arg.', '# Imp./pred.']
    if verbose >= 1:
        table_header.extend([
            '# Imp.Arg.in.range', '# Oracle', 'Oracle Recall'])
    if verbose >= 2:
        table_header.extend([
            '# Imp.Arg.0', '# Imp.Arg.1', '# Imp.Arg.2',
            '# Imp.Arg.3', '# Imp.Arg.4'])

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_cols_align(['c'] * len(table_header))
    table.set_cols_valign(['m'] * len(table_header))
    table.set_cols_width([15] * len(table_header))
    table.set_precision(2)

    table.header(table_header)
    for row in table_content:
        table.add_row(row)

    print table.draw()


def print_eval_stats(all_rich_predicates):
    predicates_by_pred = defaultdict(list)

    for rich_predicate in all_rich_predicates:
        predicates_by_pred[rich_predicate.n_pred].append(rich_predicate)

    num_dict = {}

    total_dice = 0.0
    total_gt = 0.0
    total_model = 0.0

    for n_pred, predicates in predicates_by_pred.items():
        num_dict[n_pred] = [len(predicates)]
        num_dict[n_pred].append(
            sum([predicate.num_imp_args() for predicate in predicates]))

        pred_dice = 0.0
        pred_gt = 0.0
        pred_model = 0.0
        for predicate in predicates:
            pred_dice += predicate.sum_dice
            pred_gt += predicate.num_gt
            pred_model += predicate.num_model

        total_dice += pred_dice
        total_gt += pred_gt
        total_model += pred_model

        precision, recall, f1 = compute_f1(pred_dice, pred_gt, pred_model)

        num_dict[n_pred].append(precision * 100)
        num_dict[n_pred].append(recall * 100)
        num_dict[n_pred].append(f1 * 100)

    total_precision, total_recall, total_f1 = \
        compute_f1(total_dice, total_gt, total_model)

    total_pred = 0
    total_arg = 0

    table_content = []

    for n_pred, num in num_dict.items():
        table_row = [n_pred] + num
        table_content.append(table_row)

        total_pred += num[0]
        total_arg += num[1]

    table_content.sort(key=itemgetter(2), reverse=True)
    table_content.append([''] * 6)
    table_content.append([
        'Overall',
        total_pred,
        total_arg,
        total_precision * 100,
        total_recall * 100,
        total_f1 * 100])

    table_header = [
        'Pred.', '# Pred.', '# Imp.Arg.', 'Precision', 'Recall', 'F1']

    table = Texttable()
    table.set_deco(Texttable.BORDER | Texttable.HEADER)
    table.set_precision(2)

    table.header(table_header)
    for row in table_content:
        table.add_row(row)

    print table.draw()

    print 'total_dice = {}, total_groudtruth = {}, total_model = {}'.format(
        total_dice, total_gt, total_model)
