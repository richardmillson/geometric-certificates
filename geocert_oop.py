"""  OOP refactor of geocert so I get a better feel for how the ICML
    implementation went -mj (3/1/19)
"""

from _polytope_ import Polytope, Face, from_polytope_dict
import utilities as utils
import torch
import numpy as np
import heapq
import matplotlib.pyplot as plt

##############################################################################
#                                                                            #
#                               BATCHED GEOCERT                              #
#                                                                            #
##############################################################################


class BatchGeocert(object):

    def __init__(self, polytope_list, comp_method='slow',
                 verbose=True):
        self.polytope_list = polytope_list
        for poly in self.polytope_list:
            poly.to_comparison_form()



        assert comp_method in ['slow', 'unstable', 'fast_ReLu']
        self.comp_method = comp_method

        self.verbose = verbose

    def _safety_setup(self, x ):
        """
        Just make sure this is well-posed.
        For now, return True iff x is in at least one of the polytopes

        Technically we also require the union of polytopes to be perfectly
        glued, but this is harder to check, so we leave it for now
        """
        return any(p.is_point_feasible(x) for p in self.polytope_list)


    def compute_boundaries(self):
        """ Generates a list of the shared and unshared (n-1 dimensional) faces
            of the boundary of the polytope list
        ARGS:
            None
        RETURNS:
            (unshared_facets, shared_facets) where each is a list of Face
            objects
        """
        # First gather all the feasible facets from the polytope list
        total_facets = [facet for poly in self.polytope_list for
                        facet in poly.generate_facets(check_feasible=True)]

        if self.verbose:
            print('num total facets:', len(total_facets))

        unshared_facets, shared_facets = [], []

        # Next handle techniques for which comparison method we use
        same_facet_dict = {'slow': Face.check_same_facet_pg_slow,
                             'unstable': Face.check_same_facet_pg,
                             'fast_ReLu': Face.check_same_facet_config}
        same_facet = same_facet_dict[self.comp_method]


        # Loop through all facets, figure out which are shared/unshared
        for og_facet in total_facets:
            bool_unshared = [same_facet(og_facet, ex_facet)
                             for ex_facet in unshared_facets]
            bool_shared = [same_facet(og_facet, ex_facet)
                           for ex_facet in shared_facets]

            # Don't add or remove anything already accounted for in shared list
            if any(bool_shared):
                continue

            # Remove the shared facet from the 'unshared list', add to 'shared'
            elif any(bool_unshared):
                index = bool_unshared.index(True)
                shared_facets.append(unshared_facets.pop(index))

            # Otherwise, just add this facet to the 'unshared list'
            else:
                unshared_facets.append(og_facet)

        return unshared_facets, shared_facets


    def min_dist(self, x, norm='l_2'):
        """ Returns the minimum distance from self.x to the boundary of the
            polytopes.
        ARGS:
            None
        RETURNS:
            if self.x is not contained in the list of polytopes, returns
            (-1, None, None)
            else, returns (min_dist, boundary, shared_facets) where:
            - min_dist is the minimum l_p dist to the boundary,
            - boundary is a list of facets that define the boundary
            - shared_facets is a list of facets that are shared amongst the
                            polytopes
        """
        assert norm in ['l_inf', 'l_2']

        if not self._safety_setup(x):
            return -1, None, None

        if self.verbose:
            print('----------Computing Boundary----------')
        boundary, shared_facets = self.compute_boundaries()

        dist_fxn = {'l_inf': Face.linf_dist,
                    'l_2': Face.l2_dist}[norm]

        x = x.reshape(-1, 1)
        min_dist = min(dist_fxn(facet, x) for facet in boundary)
        return min_dist, boundary, shared_facets



###############################################################################
#                                                                             #
#                               INCREMENTAL GEOCERT                           #
#                                                                             #
###############################################################################


class HeapElement(object):
    """ Wrapper of the element to be pushed around the priority queue
        in the incremental algorithm
    """
    def __init__(self, lp_dist, facet,
                 decision_bound=False,
                 exact_or_estimate='exact'):
        self.lp_dist = lp_dist
        self.facet = facet
        self.decision_bound = decision_bound
        self.exact_or_estimate = exact_or_estimate

    def __lt__(self, other):
        return self.lp_dist < other.lp_dist


class IncrementalGeoCert(object):
    def __init__(self, net, verbose=True, display=True, save_dir=None,
                 ax=None):

        # Direct input state
        self.lp_norm = None # filled in later
        self.true_label = None # filled in later
        self.lp_dist = None # filled in later
        self.net = net
        self.verbose = verbose
        self.display = display
        self.save_dir = save_dir
        self.ax = ax

        # Things to keep track of
        self.seen_to_polytope_map = {} # binary config str -> Polytope object
        self.seen_to_facet_map = {} # binary config str -> Facet list
        self.pq = [] # Priority queue that contains HeapElements


    def _verbose_print(self, *args):
        if self.verbose:
            print(*args)

    def _update_step(self, poly, popped_facet):
        """ Given the next polytope from the popped heap, does the following:
            1) Gets and stashes all the facets of the provided polytope
            2) For each facet, computes lp distance to x and adds to heap
            3) Adds this polytope's adversarial constraints to the heap too
        ARGS:
            poly : _polytope_.Polytope object - which linear region we're
                       exploring
            popped_facet : _polytope_.Face object - which face we came from
                           (because we don't want to add it into the pq again)
        RETURNS:
            None, but modifies the heap
        """

        # Step 1) Get the new facets, check their feasibility and keep track
        new_facets = poly.generate_facets_configs(self.seen_to_polytope_map,
                                                  self.net, check_feasible=True)
        self._verbose_print("Num facets: ", len(new_facets))


        poly_config = utils.flatten_config(poly.config)
        adv_constraints = self.net.make_adversarial_constraints(poly.config,
                                                                self.true_label)
        self.seen_to_polytope_map[poly_config] = poly
        self.seen_to_facet_map[poly_config] = new_facets

        # Step 2) For each new facet, add to queue
        handled_popped_facet = (popped_facet == None)
        for facet in new_facets:
            if (not handled_popped_facet) and\
                popped_facet.facet.check_same_facet_config(facet):
                handled_popped_facet = True
                continue
            facet_distance = self.lp_dist(facet, self.x)
            heap_el = HeapElement(facet_distance, facet, decision_bound=False,
                                  exact_or_estimate='exact')
            heapq.heappush(self.pq, heap_el)

        # Step 3) Adds the adversarial constraints
        for facet in adv_constraints:
            facet_distance = self.lp_dist(facet, self.x)
            heap_el = HeapElement(facet_distance, facet, decision_bound=True,
                                  exact_or_estimate='exact')
            heapq.heappush(self.pq, heap_el)

    def min_dist(self, x, lp_norm='l_2'):
        """ Returns the minimum distance between x and the decision boundary.
            Plots things too, I guess...
        """

        ######################################################################
        #   Step 0: Clear and setup state                                    #
        ######################################################################

        # Computed state
        assert lp_norm in ['l_2', 'l_inf']
        self.lp_norm = lp_norm
        self.x = x
        self.true_label = int(self.net(x).max(1)[1].item()) # classifier(x)
        self.lp_dist = {'l_2': Face.l2_dist,
                        'l_inf': Face.linf_dist}[self.lp_norm]


        # Things to keep track of
        self.seen_to_polytope_map = {} # binary config str -> Polytope object
        self.seen_to_facet_map = {} # binary config str -> Facet list
        self.pq = [] # Priority queue that contains HeapElements


        ######################################################################
        #   Step 1: handle the initial polytope                              #
        ######################################################################
        self._verbose_print('---Initial Polytope---')
        p_0_dict = self.net.compute_polytope(self.x, True)
        p_0 = from_polytope_dict(p_0_dict)
        self._update_step(p_0, None)

        ######################################################################
        #   Step 2: Repeat until we hit a decision boundary                  #
        ######################################################################

        index = 0
        while True:
            pop_el = heapq.heappop(self.pq)
            # If popped el is part of decision boundary, we're done!
            if pop_el.decision_bound:
                self._verbose_print('----------Minimal Projection Generated----------')
                self.plot_2d(pop_el.lp_dist, iter=index)
                return pop_el.lp_dist

            # Otherwise, open up a new polytope and explore
            else:
                self._verbose_print('---Opening New Polytope---')

                popped_facet = pop_el.facet
                configs = popped_facet.get_new_configs(self.net)
                configs_flat = utils.flatten_config(configs)


                # If polytope has already been seen, don't add it again
                if configs_flat not in self.seen_to_polytope_map:
                    new_poly_dict = self.net.compute_polytope_config(configs,
                                                                     True)
                    new_poly = from_polytope_dict(new_poly_dict)
                    self._update_step(new_poly, pop_el)
                else:
                    self._verbose_print("We've already seen that polytope")

            if index % 1 == 0 and self.display:
                self.plot_2d(pop_el.lp_dist, iter=index)
            index += 1



    def plot_2d(self, t, n_colors=200, iter=0):
        ''' Plots the current search boundary based on the heap, the seen polytopes,
            the current minimal lp ball, and any classification boundary facets
        '''
        plt.figure(figsize=[10, 10])
        ax = self.ax if self.ax is not None else plt.axes()

        polytope_list = [self.seen_to_polytope_map[elem]
                         for elem in self.seen_to_polytope_map]
        facet_list = [heap_elem.facet for heap_elem in self.pq
                      if not heap_elem.decision_bound]
        boundary_facet_list = [heap_elem.facet for heap_elem in self.pq
                               if heap_elem.decision_bound]
        colors = utils.get_spaced_colors(n_colors)[0:len(polytope_list)]

        xylim = 50.0
        utils.plot_polytopes_2d(polytope_list, colors=colors, alpha=0.7,
                              xylim=5, ax=ax, linestyle='dashed', linewidth=0)

        utils.plot_facets_2d(facet_list, alpha=0.7,
                             xylim=xylim, ax=ax, linestyle='dashed',
                            linewidth=3, color='black')

        utils.plot_facets_2d(boundary_facet_list, alpha=0.7,
                             xylim=xylim, ax=ax, linestyle='dashed',
                             linewidth=3, color='red')

        if self.lp_norm == 'l_inf':
            utils.plot_linf_norm(self.x, t, linewidth=1, edgecolor='red', ax=ax)
        elif self.lp_norm == 'l_2':
            utils.plot_l2_norm(self.x, t, linewidth=1, edgecolor='red', ax=ax)
        else:
            raise NotImplementedError

        plt.autoscale()
        new_xlims = plt.xlim()
        new_ylims = plt.ylim()


        if (min(new_xlims) > -xylim) and \
           (max(new_xlims) < xylim) and \
           (min(new_ylims) > -xylim) and \
           (max(new_ylims) < xylim):
            pass
        else:
            plt.xlim(-xylim, xylim)
            plt.ylim(-xylim, xylim)
        if self.save_dir is not None:
            filename = self.save_dir + str(iter) + '.png'
            plt.savefig(filename)
        plt.close()



