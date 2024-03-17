__author__ = 'modlab'

import sys
import collections 
if sys.version_info.major == 3 and sys.version_info.minor >= 10:
    from collections.abc import MutableMapping
else:
    from collections import MutableMapping


class PartialChargeDict(MutableMapping):
    """ Dictionary Decorator to store Partial Charges

    Class internally stores an OrderedDict with different
    partial charges. The access to them is controlled, i.e.
    only known (PartialCharges.defined_partial_charges)
    are allowed as keys. Some additional helper functions
    should ease the access.

    Help taken from: http://stackoverflow.com/questions/3387691/python-how-to-perfectly-override-a-dict
    """

    defined_partial_charges = ('Mulliken', 'mulliken', 'hirshfeld', 'cm5', 'esp', 'npa', 'dftb', 'am1')
    _representation = {'mulliken': 'Mulliken',
                       'hirshfeld': 'Hirshfeld',
                       'cm5': 'CM5',
                       'esp': 'ESP',
                       'npa': 'NPA',
                       'dftb': 'DFTB',
                       'am1': 'AM1'
                       }

    def __init__(self, *args, **kwargs):
        # Main container to store the associated values
        self.store = collections.OrderedDict()
        self.update(collections.OrderedDict(*args, **kwargs))

    def __getitem__(self, item):
        return self.store.__getitem__(item)

    def __setitem__(self, key, value):
        if key not in self.defined_partial_charges:
            raise Exception('Partial charge not known: {}'.format(key))
        self.store[key] = value

    def __delitem__(self, key):
        del self.store[key]

    def __iter__(self):
        return iter(self.store)

    def __len__(self):
        return len(self.store)

    def represent(self, name):
        assert name in self.defined_partial_charges, 'Requested partial charge is not known'
        return self._representation[name]

    def iteritems(self):
        return self.store.iteritems()
