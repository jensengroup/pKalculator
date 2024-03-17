from DescriptorCreator.DescriptorElement import DescriptorElement
from DescriptorCreator.PartialChargeDict import PartialChargeDict as PCD
import logging

__author__ = 'modlab'

def adjacent_pairs(seq):
    it = iter(seq)
    a = it.__next__()
    for b in it:
        yield a, b
        a = b


class GraphChargeShell(DescriptorElement):

    description = 'Charge shell based on sorted molecular graph'

    def __init__(self, **options):
        super(GraphChargeShell, self).__init__()
        self.n_shells = options.pop('n_shells')
        self.charge = options.pop('charge_type')
        self.cip = options.pop('use_cip_sort', False) # USE Cahn Ingold Prelog Type rules for sorting
        self._current_mol = None

    def calculate_elements(self, atom):
        charge = self.charge
        self._current_mol = atom.GetOwningMol()
        last_shell = []
        current_shell = []
        block = []  # Corresponds to a block of atoms which then have to be sorted, could be sourced to a own class

        # set the atomic charge as the first element
        descriptor_elements = [float(atom.GetProp(charge))]
        mapper = [atom.GetIdx()]
        contained_atoms = set()  # !! Indices need to be stored because obviously rdkit does not return references to same object
        contained_atoms.add(atom.GetIdx())
        last_shell.append(atom)

        length = 4  # first atom has 4 neighbours
        for shell in range(self.n_shells):
            for ls_atom in last_shell:
                # loop through shell and create next shell
                # create and sort subshells / blocks
                for neighbour_atom in ls_atom.GetNeighbors():
                    # NOTE !!!: Identity comparison seemingly does not work as expected. I do not now if he checks pointer
                    # identity or if the object is copied unknown, which both might happen and leads to undesired results
                    # and obviously does not work. Everytime i assign an atom with a =mol.GetAtoms()[0] it becomes a
                    # different address in memory. Thus I have to compare indices ....
                    if neighbour_atom.GetIdx() not in contained_atoms:
                        block.append(neighbour_atom)
                # Fill the block with dummy atoms, such that all blocks have the same length
                self._fill_block(block, length)
                # Sort the block according to the chosen sorting scheme
                if self.cip:
                    self._sort_block_cip(block)
                else:
                    self._sort_block(block)
                current_shell = current_shell + block
                block = []
                length = 3  # from the second shell on atoms only have 3 members because one is already contained

                indices_block = [atm.GetIdx() for atm in current_shell]
                contained_atoms.update(indices_block)

            indices = [atm.GetIdx() for atm in current_shell]
            #contained_atoms.update(indices)
            descriptor_elements += [float(atm.GetProp(charge)) for atm in current_shell]
            mapper += indices
            last_shell = current_shell
            current_shell = []
        assert len(descriptor_elements) == self._calculate_length(self.n_shells), '{}  {}'.format(len(
            descriptor_elements),  self._calculate_length(self.n_shells))  # Debugging assertion
        
        #print('REMEMBER TO REFIX BLOCK!!!')
        return descriptor_elements, mapper

    @classmethod
    def _calculate_length(cls, n):
        result = 5
        previous_shell = 4
        m = 1
        while m < n:
            result += 3*previous_shell
            previous_shell = 3*previous_shell
            m += 1
        return result

    def _fill_block(self, block, length):
        """ Fills block with dummy atoms to length and than further sends it to the sorting routine
        :param block: block containing the atoms to sort
        :param length: the length to which the block should be filled with dummy atoms
        :return:
        """

        class DummyA(object):
            def __init__(self):
                self.charge = 0.0

            @classmethod
            def GetNeighbors(cls):
                return []

            @classmethod
            def GetIdx(cls):
                return -1

            def GetProp(self, charge):
                props = list(PCD.defined_partial_charges) + ['f_n', 'f_e', 'f_r']
                assert charge in props
                return self.charge

            @classmethod
            def GetAtomicNum(cls):
                return -10

        assert len(block) <= length, 'There is a atom with more than 4 neighbours!\n{}'.format(block)
        while len(block) < length:
            block.append(DummyA())

    def _sort_block(self, block):
        """ Sort the block according to partial charges only
        :param block: Block of Atoms that need to be sorted (type: list)
        :type block: list
        :return: None
        """
        block.sort(key=lambda a: float(a.GetProp(self.charge)))

    def _sort_block_cip(self, block):
        #TODO: Still has to be broken up in smaller functional chuncks. I.E. general helper methods for set extraction
        """ Sort block according to CIP rules and charges if CIP is unambiguous

        Sorting according to the CIP rules works as follows:
        1) Take all bound substituents and sort according to MW.
        2) If 1 is not unique, for each atom with same priority (A*):
            a) Go to bound and yet not included atoms and sum up MWs. Set priority of A* according to summed MW
            b) If 2 a) did not give unambiguous result expand shell of each atom A* by one bond.
            c) repeat 2b) until unique order is found
            d) If no unique order is found and all bound atoms are included (in set of already summed in atoms)
               sort atoms according to charge (this is an arbitrary choice)

        :param block: Block of Atoms that need to be sorted
        :type block: list
        :return: None
        """
        molecule = self._current_mol
        logging.debug('Entering CIP determination routine')
        logging.debug('length of block: {}'.format(len(block)))
        # The atom that needs to be determined can safely be ignored, because the same mass is added
        # For each of its substituents if they "follow the bond to the origin"
        # This is the simplest but not most efficient implementation
        priorities = [atm.GetAtomicNum() for atm in block]
        logging.debug('initial priorities: {}'.format(priorities))
        # For each tbd. atom a list of neighbors per shell (defining 'rest') to calc MW of rest from
        # The atoms have to be stored as their idx because RDkit atom comparison does not work! --> RDkit issue ...
        neighbour_sets = [{atm_x.GetIdx()} for atm_x in block]
        while len(priorities) != len(set(priorities)):
            old_set_caridnalities = [len(s) for s in neighbour_sets]
            logging.debug(' old_set_cardinalities: {}'.format(old_set_caridnalities))
            # In each loop the shell is enlarged by one bond
            for num, atom in enumerate(block):
                # add the neighbor one bond apart to the set to calculate the MW from
                for nbr in atom.GetNeighbors():
                    neighbour_sets[num].add(nbr.GetIdx())
            for num, neigbor_idx_set in enumerate(neighbour_sets):
                priorities[num] = sum([molecule.GetAtoms()[index].GetAtomicNum() for index in neighbour_sets[num]])
            # Check new set cardinalities. If they did not change the loop needs to be aborted
            new_set_cardinalities = [len(s) for s in neighbour_sets]
            if new_set_cardinalities == old_set_caridnalities:
                break
        # sort block according to priorities so that the block is synchronous with sorted priorities that
        # the next step can be conducted
        block[:] = [elem for (prior, elem) in sorted(zip(priorities, block), key=lambda x:x[0])]
        priorities.sort()
        # Note that dummy atoms end up at the end of a block and need no further attendance because they will anyways
        # only produce zeroes in their sub graph

        # Now check if all priorities are different and if not sort those with same priors acc to charge
        # For this successive items of the priorities list have to be compared
        logging.debug('priorities: {}'.format(priorities))
        if len(priorities) != len(set(priorities)):
            idx = 0
            # Set containing mappings (actual, target) for each entry to reorganize the list
            mappings = dict()
            sets_to_sort = []
            indices_of_group_to_sort = set()
            for ele, next_ele in adjacent_pairs(priorities):
                if ele == next_ele:
                    if idx in indices_of_group_to_sort:
                        indices_of_group_to_sort.add(idx+1)
                    else:
                        indices_of_group_to_sort.add(idx)
                        indices_of_group_to_sort.add(idx+1)
                else:
                    mappings[idx] = idx
                    if len(indices_of_group_to_sort) > 0:
                        sets_to_sort.append(indices_of_group_to_sort)
                        indices_of_group_to_sort = set()
                idx += 1
            if len(indices_of_group_to_sort) != 0:
                sets_to_sort.append(indices_of_group_to_sort)
            logging.debug('MAPPINGS 1 : {}'.format(mappings))
            # The las index needs special treatment. If it was not added to mapping in step where two successive
            # pairs were compared it needs to be added by hand
            if not len(block)-1 in mappings:
                mappings[len(block)-1] = len(block)-1

            # sort sets that have to be sorted according to charge
            logging.debug('Sets to sort:  {}'.format(sets_to_sort))
            for index_set in sets_to_sort:
                logging.debug('Index set: {}'.format(index_set))
                orig_list = list(index_set)
                sorted_list = sorted(orig_list, key=lambda a: float(block[a].GetProp(self.charge)))
                combined = list(zip(orig_list, sorted_list))
                for idx1, idx2 in combined:
                    mappings[idx1] = idx2
                    logging.debug('MAPPINGS 2 : {}'.format(mappings))
            # check that the mapping has as many elements as the original block
            assert len(mappings) == len(block), 'mappings = {} ;;;; block = {}'.format(mappings, block)
            # apply mapping to create new block
            sorted_block = [0 for elem in block]
            for key, val in mappings.items():
                new_idx = val
                old_idx = key
                sorted_block[new_idx] = block[old_idx]
            assert len(sorted_block) == len(block)
            block[:] = sorted_block  # Note that just assigning does not work!
