nominal_predicate_mapping = {
    'bid': 'bid',
    'sale': 'sell',
    'loan': 'loan',
    'cost': 'cost',
    'plan': 'plan',
    'investor': 'invest',
    'price': 'price',
    'loss': 'lose',
    'investment': 'invest',
    'fund': 'fund',
}

predicate_core_arg_mapping = {
    'bid': {
        'arg0': 'SUBJ',
        'arg1': 'PREP_for',
        'arg2': 'OBJ'
    },
    'sell': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP_for',
        'arg4': 'PREP'
    },
    'loan': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP',
        'arg4': 'PREP_at'
    },
    'cost': {
        'arg1': 'SUBJ',
        'arg2': 'OBJ',
        'arg3': 'PREP_to',
        'arg4': 'PREP'
    },
    'plan': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_for',
        'arg3': 'PREP_for'
    },
    'invest': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_in'
    },
    'price': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_at',
        'arg3': 'PREP'
    },
    'lose': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP_to',
        'arg3': 'PREP_on'
    },
    'fund': {
        'arg0': 'SUBJ',
        'arg1': 'OBJ',
        'arg2': 'PREP',
        'arg3': 'PREP'
    },
}

nombank_function_tag_mapping = {
    'TMP': 'temporal',
    'LOC': 'location',
    'MNR': 'manner',
    'PNC': 'purpose',
    'NEG': 'negation',
    'EXT': 'extent',
    'ADV': 'adverbial',
    # tags below do not appear in implicit argument dataset
    'DIR': 'directional',
    'PRD': 'predicative',
    'CAU': 'cause',
    'DIS': 'discourse',
    'REF': 'reference'
}

pred_list = [
    'bid', 'sell', 'loan', 'cost', 'plan', 'invest', 'price', 'lose', 'fund']

core_arg_list = ['arg0', 'arg1', 'arg2', 'arg3', 'arg4']


def expand_wsj_fileid(fileid, ext=''):
    assert fileid[:4] == 'wsj_' and fileid[4:].isdigit()
    return fileid.split('_')[1][:2] + '/' + fileid + ext


def shorten_wsj_fileid(fileid):
    result = fileid[3:11]
    assert result[:4] == 'wsj_' and result[4:].isdigit()
    return result


def convert_nombank_label(label):
    if label[:3] == 'ARG':
        if label[3].isdigit():
            return label[:4].lower()
        elif label[3] == 'M':
            function_tag = label.split('-')[1]
            return nombank_function_tag_mapping.get(function_tag, '')
    return ''


def compute_f1(total_dice, total_gt, total_model):
    if total_model != 0.0:
        precision = total_dice / total_model
    else:
        precision = 1
    if total_gt != 0.0:
        recall = total_dice / total_gt
    else:
        recall = 1
    if precision != 0.0 and recall != 0.0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    return precision, recall, f1
