__all__ = ['atomic_num_to_element', 'element_to_atomic_num', 'rn_to_num', 'num_to_rn']

_elements = [
    'H',  'He', 'Li', 'Be', 'B',  'C',  'N',  'O',  'F',  'Ne',
    'Na', 'Mg', 'Al', 'Si', 'P',  'S',  'Cl', 'Ar', 'K',  'Ca',
    'Sc', 'Ti', 'V',  'Cr', 'Mn', 'Fe', 'Co', 'Ni', 'Cu', 'Zn',
    'Kr', 'Mo', 'Xe'
]
_anums = list(range(1, 31)) + [36, 42, 54]
_anum_elem_mapping = dict(zip(_anums, _elements))
_elem_anum_mapping = dict(map(reversed, _anum_elem_mapping.items()))

def atomic_num_to_element(num):
    """Return the symbol for the element with atomic number `num`."""
    try:
        return _anum_elem_mapping[num]
    except KeyError as e:
        raise ValueError(f"Element number {num} is not implemented") from e

def element_to_atomic_num(elem):
    """Return the symbol for the element with atomic number `num`."""
    try:
        return _elem_anum_mapping[elem]
    except KeyError as e:
        raise ValueError(f"Element {elem} is not recognised") from e

_dec_rn_mapping = dict(zip([1, 4, 5, 9, 10, 40, 50, 90],
                           ['I', 'IV', 'V', 'IX', 'X', 'XL', 'L', 'XC']))
_rn_dec_mapping = dict(map(reversed, _dec_rn_mapping.items()))

def rn_to_num(numeral, sub_one=True):
    """
    Convert a roman numeral to the corresponding decimal number.
    If `sub_one` is True, the number returned is one less than the directly equivalent value.
    This option should be set if the number is intended to repesent a charge number (eg H^0),
    and should be left False if it is intended to represent a species number (eg HI).
    """
    if type(numeral) is not str:
        raise ValueError(f"Expected str for type(num), got {type(numeral)}")

    if sub_one:
        num = -1
    else:
        num = 0
    
    try:
        while numeral:
            highnum = max(filter(lambda num: numeral.startswith(num), _rn_dec_mapping),
                          key=lambda v: _rn_dec_mapping[v])
            num += _rn_dec_mapping[highnum]
            numeral = numeral.replace(highnum, '', 1)
    except ValueError as e:
        raise ValueError("Values above 99 are not implemented") from e
    return num

def num_to_rn(num, add_one=False):
    """
    Convert a decimal number to the corresponding roman numeral.
    If `sub_one` is True, the numeral returned is one more than the directly equivalent value.
    This option should be set if the provided `num` repesents a charge number (eg H^0),
    and should be left False if it represents a species number (eg HI).
    """
    if type(num) is not int:
        raise ValueError(f"Expected int for type(num), got {type(num)}")  
    if num > 99:
        raise ValueError("Values above 99 are not implemented")

    if add_one:
        num += 1

    rn = ''
    while num > 0:
        highval = max(filter(lambda x: x <= num, _dec_rn_mapping))
        rn += _dec_rn_mapping[highval]
        num -= highval
    return rn
