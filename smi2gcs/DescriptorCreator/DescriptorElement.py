__author__ = 'modlab'


class DescriptorElement(object):
    """ Abstract base class for a descriptor property. It sets the interface that should be followed.
    Each property should have a understandable description.
    The initializer of child classes should be called with super() and options forwarded to
    guarantee well behaved options handling. calculate_elements() must be implemented to return
    a list of descriptor elements.
    """
    description = 'To be filled with property description'

    def __init__(self, **options):
        if options:
            raise Exception('Unhandeled options : {}'.format(options))

    def calculate_elements(self, described_object):
        """
        :param described_object: The described_object for which the descriptor element is calculated
        :return: list with descriptor elements
        """
        raise NotImplementedError