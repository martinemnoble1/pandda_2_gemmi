import numpy as np
from scipy import spatial
from scipy.cluster.hierarchy import fclusterdata
import gemmi
from rich import print as rprint
from rich.traceback import install
install(show_locals=True)
import scipy
import matplotlib.pyplot as plt

# import networkx as nx


from ..interfaces import *
from pandda_gemmi.dataset import StructureArray
from pandda_gemmi.dataset.structure import is_protein_residue
from pandda_gemmi.processor import Partial
from pandda_gemmi.alignment import Alignment

from .site import Site


# class HeirarchicalSiteModel:
#
#     def __init__(self, t=8.0):
#         self.t = t
#
#     def __call__(self,
#          datasets: Dict[str, DatasetInterface],
#          events: Dict[Tuple[str, int], EventInterface],
#          processor: ProcessorInterface,
#          structure_array_refs,
#                  ):
#
#         #
#         if len(events) == 0:
#             return {}
#
#         if len(events) == 1:
#             return {0: Site(
#                 list(events.keys())[0],
#                 np.mean(list(events.values())[0].pos_array, axis=1)
#             )}
#
#         # Get the array of centroids
#         centroid_array = np.array(
#             [
#                 np.mean(event.pos_array, axis=0)
#                 for event
#                 in events.values()
#             ]
#         )
#
#         #
#         clusters = fclusterdata(
#             centroid_array,
#             t=self.t,
#             criterion="distance",
#             method="centroid"
#         )
#
#         #
#         unique_clusters = np.unique(clusters, )
#
#         # Merge the events based on the clustering
#         event_array = np.array([event_id for event_id in events.keys()])
#         j = 0
#         sites = {}
#         for cluster_num in unique_clusters:
#             site_event_ids = event_array[clusters == cluster_num]
#             # print([event_id for event_id in site_event_ids])
#             site_positions = np.concatenate([events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)
#
#             sites[j] = Site(
#                 [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
#                 np.mean(site_positions, axis=0),
#             )
#             j += 1
#
#         return sites
# def _get_closest_transform(unaligned_centroid, transforms, com_ref, com_mov):
#     distances = []
#     for transform, com_ref_centroid, com_mov_centroid in zip(transforms.values(), com_ref.values(), com_mov.values()):
#         distances.append(
#             gemmi.Position(*com_mov_centroid).dist(gemmi.Position(*unaligned_centroid))
#         )
#
#     min_dist_index = np.argmin(distances)
#     return list(transforms.values())[min_dist_index], list(com_ref.values())[min_dist_index], list(com_mov.values())[min_dist_index]


class HeirarchicalSiteModel:

    def __init__(self, t=8.0, debug=False):
        self.t = t
        self.debug = debug

    def get_event_environment(
            self,
            query_pos_array,
            structure_array,
            ns
    ):
        indexess = ns.query_ball_point(
            query_pos_array,
            5.0,
        )
        res_ids = {}

        for indexes in indexess:
            chains = structure_array.chains[indexes]
            residues = structure_array.seq_ids[indexes]

            for chain, res in zip(chains, residues):
                res_ids[(str(chain), str(res))] = True

        return [res_id for res_id in res_ids.keys()]


    def get_event_environments(self, datasets, events):
        # Get the environments for each event
        event_evenvironments = {}
        for dtag, dataset in datasets.items():
            st_arr = StructureArray.from_structure(dataset.structure)
            ns = spatial.KDTree(st_arr.positions)
            dtag_events = {event_id: event for event_id, event in events.items() if event_id[0] == dtag}


            for event_id, event in dtag_events.items():
                event_evenvironments[event_id] = self.get_event_environment(
                    event.pos_array,
                    st_arr,
                    ns,
                )

        return event_evenvironments

    def get_centroids(self, event_environments, reference_dataset):
        centroids = {}
        for event_id, environment in event_environments.items():
            # res_ids = set([res_id for _event_id in clique for res_id in event_environments[_event_id]])
            poss = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for model in reference_dataset.structure.structure
                for chain in model
                for res in chain
                for atom in res
                if (chain.name, str(res.seqid.num)) in environment
            ]
            if len(poss) == 0:
                centroid = [0.0,0.0,0.0]
            else:
                centroid = np.mean(
                    np.array(poss),
                    axis=0
                )
            centroids[event_id] = centroid

        return centroids

    def __call__(self,
                 datasets: Dict[str, DatasetInterface],
                 events: Dict[Tuple[str, int], EventInterface],
                 ref_dataset,
                 existing_events,
                 existing_sites
                 ):

        # Handle edge cases
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site(
                list(events.keys()),
                np.mean(list(events.values())[0].pos_array, axis=0)
            )}

        # Find the residue environment of each event (chain and residue number)
        event_environments: Dict[Tuple[str, int], List[Tuple[str, str]]] = self.get_event_environments(datasets, events)

        # Find the site centroids against the reference
        centroids = self.get_centroids(
            event_environments,
            ref_dataset,
        )

        # Cluster the centroids
        event_id_array = np.array(
            [_event_id for _event_id in centroids.keys()]
        )
        centroid_array = np.array(
            [
                centroid
                for centroid
                in centroids.values()
            ]
        )
        clusters = fclusterdata(
            centroid_array,
            t=self.t,
            criterion="distance",
            method="centroid"
            # method="complete"
        )

        # Get the event sites
        event_sites = {}
        for j, cluster in enumerate(np.unique(clusters)):
            cluster_event_id_array = event_id_array[clusters == cluster]
            for event_id in cluster_event_id_array:
                event_sites[(str(event_id[0]), int(event_id[1]))] = j

        sites = {}

        # If there are existing sites, first construct these clusters, including any new datasets
        # that cluster with old ones. Keep any known events in their sites regardless of new clustering.
        allocated_events = []
        if existing_sites:
            allocated_events = [
                (_row['dtag'], int(_row['event_idx']))
                 for _row
                 in existing_events.values()
            ]
            for site_idx, site_info in existing_sites.items():
                # Get known events in this site
                known_site_events = [
                    (_row['dtag'], int(_row['event_idx']))
                     for _row
                     in existing_events.values()
                     if _row['site_idx'] == site_idx
                ]

                # Get any new datasets that cluster with these (and aren't in a known site)
                known_site_events_new_sites = set(
                    [
                        new_site_idx
                        for event_id, new_site_idx
                        in event_sites.items()
                        if event_id in known_site_events
                    ]
                )
                new_overlapping_events = [
                    new_event_id
                    for new_event_id, new_site_idx
                    in event_sites.items()
                    if (new_site_idx in known_site_events_new_sites) & (new_event_id not in existing_events) & (new_event_id not in allocated_events)
                ]

                sites[site_idx] = Site(
                    known_site_events + new_overlapping_events,
                    site_info['centroid'],
                    site_info['Name'],
                    site_info['Comment']
                )
                # Allocate new events that have been used
                for _event_id in new_overlapping_events:
                    allocated_events.append(_event_id)

        if self.debug:
            print('event_id_array')
            print(event_id_array)
            print('clusters')
            print(clusters)

        # Then add new sites for any events that haven't already been explained.
        # Construct the sites
        for cluster in np.unique(clusters):
            cluster_event_id_array = event_id_array[clusters == cluster]

            # Get new, unallocated events
            new_site_events = [
                (str(event_id[0]), int(event_id[1]))
                for event_id
                in cluster_event_id_array
                if (str(event_id[0]), int(event_id[1])) not in allocated_events
            ]

            if len(new_site_events) != 0:

                sites[len(sites)+1] = Site(
                    new_site_events,
                    np.mean(
                        centroid_array[clusters==cluster, :],
                        axis=0,
                    ).flatten(),
                )

        return sites

class HeirarchicalSiteModelAlignedSequences:

    def __init__(self, t=0.3, debug=False, distance=8.0):
        self.t = t
        self.debug = debug
        self.distance = distance

    def get_event_environment(
            self,
            query_pos_array,
            structure_array,
            ns,
            distance = 5.0
    ):
        indexess = ns.query_ball_point(
            query_pos_array,
            distance,
        )
        res_ids = {}

        for indexes in indexess:
            chains = structure_array.chains[indexes]
            residues = structure_array.seq_ids[indexes]

            for chain, res in zip(chains, residues):
                res_ids[(str(chain), str(res))] = True

        return [res_id for res_id in res_ids.keys()]


    def get_event_environments(self, datasets, events, distance=5.0):
        # Get the environments for each event
        event_evenvironments = {}
        for dtag, dataset in datasets.items():
            st_arr = StructureArray.from_structure(dataset.structure)
            ns = spatial.KDTree(st_arr.positions)
            dtag_events = {event_id: event for event_id, event in events.items() if event_id[0] == dtag}


            for event_id, event in dtag_events.items():
                event_evenvironments[event_id] = self.get_event_environment(
                    event.pos_array,
                    st_arr,
                    ns,
                    distance=distance,
                )

        return event_evenvironments

    def get_centroids(self, event_environments, reference_dataset):
        centroids = {}
        for event_id, environment in event_environments.items():
            # res_ids = set([res_id for _event_id in clique for res_id in event_environments[_event_id]])
            poss = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for model in reference_dataset.structure.structure
                for chain in model
                for res in chain
                for atom in res
                if (chain.name, str(res.seqid.num)) in environment
            ]
            if len(poss) == 0:
                centroid = [0.0,0.0,0.0]
            else:
                centroid = np.mean(
                    np.array(poss),
                    axis=0
                )
            centroids[event_id] = centroid

        return centroids
    
    def get_insertion_mapping(self, ref_seq, ref_insertions, sequence, insertions):
        # Get the alignment
        alignment = gemmi.align_string_sequences(
            ref_seq,
            sequence,
            [],
            gemmi.AlignmentScoring('b'),
        )

        # Get the gap strings
        formatted_match = alignment.formatted(
            ''.join(ref_seq),
            ''.join(sequence),
        )
        aligned_ref, match_string, aligned_mov = formatted_match.splitlines()

        # Get native index to aligned index for the ref seq
        count = 0
        ref_to_aligned = {}
        for j, item in enumerate(aligned_ref):
            if item not in ['-', ' ']:
                ref_to_aligned[count] = j
                count += 1

        # Get the aligned index to native index for the mov seq
        mov_seq_index_to_key = {_j: _key for _j, _key in enumerate(insertions)}
        aligned_to_mov = {}
        count = 0
        for aligned_index, item in enumerate(aligned_mov):
            if item not in ['-', ' ']:
                aligned_to_mov[aligned_index] = mov_seq_index_to_key[count]
                count += 1

        # Generate the residue mapping
        residue_mapping = {}

        for ref_seqid_index, ref_seqid in enumerate(ref_insertions):
            # Get the alignment space index
            alignment_index = ref_to_aligned[ref_seqid_index]

            # Get the corresponding mov res at that alignment index
            mov_seq_res = aligned_mov[alignment_index]

            # Only consider matched residues
            if mov_seq_res != '-':
                # Get the mov seq index
                mov_seqid = aligned_to_mov[alignment_index]

                # Update the residue mapping
                residue_mapping[ref_seqid] = mov_seqid


        return residue_mapping, len(residue_mapping) / len(ref_insertions)
        ...

    def chain_to_seq(self, chain):
        seq = []
        insertion = []
        for res in chain:
            if is_protein_residue(res):
                seq.append(gemmi.one_letter_code([res.name]))
                insertion.append(str(res.seqid.num))

        return seq, insertion

    def chain_is_protein(self, chain):
        if any([is_protein_residue(res) for res in chain]):
            return True
        else:
            return False

    def get_alignments(self, datasets: Dict[str, DatasetInterface]):
        # Iterate over chains in datasets, creating alignment classes for acceptably good alignments as necessary
        alignments = {}
        for dtag in sorted(datasets, key = lambda _dtag: datasets[_dtag].reflections.resolution()):
            for chain in datasets[dtag].structure.structure[0]:
                # If chain is non protein, skip
                if not self.chain_is_protein(chain):
                    continue

                # Get the seqeunce
                sequence, insertions = self.chain_to_seq(chain)
                    
                # Iterate over alignment references, getting alignments
                matched = False
                for ref_dtag, ref_chain in alignments:
                    ref_seq, ref_insertions = self.chain_to_seq(datasets[ref_dtag].structure.structure[0][ref_chain])
                    insertion_mapping, score = self.get_insertion_mapping(ref_seq, ref_insertions, sequence, insertions)

                    if (ref_dtag, ref_chain) == (dtag, chain.name):
                        assert score > 0.8

                    # If the alignment is "good", add to class
                    if score > 0.8:
                        matched = True
                        break

                    # Otherwise create a new alignment class and add it
                if matched:
                    alignments[(ref_dtag, ref_chain)][(dtag, chain.name)] = insertion_mapping

                else:
                    alignments[(dtag, chain.name)] = {(dtag, chain.name): self.get_insertion_mapping(sequence, insertions, sequence, insertions)[0]}
                
        return alignments
    
    def get_event_distance(self, ref_event_env, mov_event_env, msa, ref_dtag, mov_dtag):    
        # Get the msa classes for each dtag's chains
        ref_chain_classes = {}
        mov_chain_classes = {}
        for (alignment_dtag, alignment_chain), alignments in msa.items():
            for (aligned_dtag, aligned_chain), alignment in alignments.items():
                if aligned_dtag == ref_dtag:
                    ref_chain_classes[aligned_chain] = (alignment_dtag, alignment_chain)
                if aligned_dtag == mov_dtag:
                    mov_chain_classes[aligned_chain] = (alignment_dtag, alignment_chain)

        # Map residues between the environments
        try:
            ref_aligned_resids = []
            for chain, res in ref_event_env:
                if chain not in ref_chain_classes:
                    continue
                chain_class = ref_chain_classes[chain]
                alignment = msa[chain_class][(ref_dtag, chain)]
                aligned_index = alignment[res]
                ref_aligned_resids.append((chain_class[1], aligned_index))
            ref_aligned_resids_set = set(ref_aligned_resids)
        except:
            rprint('mov res and alignment')
            rprint(res)
            rprint(alignment)
            raise Exception
        
        mov_aligned_resids = []
        try:
            for chain, res in mov_event_env:
                if chain not in mov_chain_classes:
                    continue
                chain_class = mov_chain_classes[chain]
                alignment = msa[chain_class][(mov_dtag, chain)]
                aligned_index = alignment[res]
                mov_aligned_resids.append((chain_class[1], aligned_index))
        except:
            rprint('mov res and alignment')
            rprint(res)
            rprint(alignment)
            raise Exception
        mov_aligned_resids_set = set(mov_aligned_resids)

        union = mov_aligned_resids_set.union(ref_aligned_resids_set)
        intersection = mov_aligned_resids_set.intersection(ref_aligned_resids_set)

        # if len(union) == 0:
        #     return 1.0

        # else:
        #     return 1 - (len(intersection) / len(union))
        if len(intersection) == 0:
            return 1.0
        elif len(intersection) == len(union):
            return 0.0
        else:
            return 1/len(intersection)

    def get_event_distances(self, event_environments, msa):
        correlations = {}
        for ref_event_id, ref_event_env in event_environments.items():
            correlations[ref_event_id] = {}
            for mov_event_id, mov_event_env in event_environments.items():
                ref_dtag, mov_dtag = ref_event_id[0], mov_event_id[0]
                correlations[ref_event_id][mov_event_id] = self.get_event_distance(
                    ref_event_env, 
                    mov_event_env, 
                    msa, 
                    ref_dtag, 
                    mov_dtag,
                    )

        return correlations
        ...

    def __call__(self,
                 datasets: Dict[str, DatasetInterface],
                 events: Dict[Tuple[str, int], EventInterface],
                 ref_dataset,
                 existing_events,
                 existing_sites,
                 
                 ):

        # Handle edge cases
        if len(events) == 0:
            return {}

        if len(events) == 1:
            return {0: Site(
                list(events.keys()),
                np.mean(list(events.values())[0].pos_array, axis=0)
            )}
        
        # Get the sequence alignments
        msa = self.get_alignments(datasets)

        # Find the residue environment of each event (chain and residue number)
        event_environments: Dict[Tuple[str, int], List[Tuple[str, str]]] = self.get_event_environments(datasets, events, distance=self.distance)

        # Get overlaps
        distances = self.get_event_distances(event_environments, msa)

        # Find the site centroids against the reference
        distance_matrix = np.array(
            [
                [
                    distance
                    for event_id_2, distance
                    in event_distances.items()
                ]
                for event_id_1, event_distances
                in distances.items()

            ]
        )

        event_id_array = np.array(
            [_event_id for _event_id in distances.keys()]
        )

        # Cluster the centroids
        linkage = scipy.cluster.hierarchy.linkage(
            scipy.spatial.distance.squareform(distance_matrix),
            method='complete',
            optimal_ordering=True
            # metric='chebyshev'
        )
        clusters = scipy.cluster.hierarchy.fcluster(
            linkage,
            t=self.t,
            criterion="distance",
        )
        # clusters = fclusterdata(
        #     ,
            
        #     # method="centroid"
        #     method="complete",
            
        # )

        # Get the event sites
        event_sites = {}
        for j, cluster in enumerate(np.unique(clusters)):
            cluster_event_id_array = event_id_array[clusters == cluster]
            for event_id in cluster_event_id_array:
                event_sites[(str(event_id[0]), int(event_id[1]))] = j

        sites = {}

        # If there are existing sites, first construct these clusters, including any new datasets
        # that cluster with old ones. Keep any known events in their sites regardless of new clustering.
        allocated_events = []
        if existing_sites:
            allocated_events = [
                (_row['dtag'], int(_row['event_idx']))
                 for _row
                 in existing_events.values()
            ]
            for site_idx, site_info in existing_sites.items():
                # Get known events in this site
                known_site_events = [
                    (_row['dtag'], int(_row['event_idx']))
                     for _row
                     in existing_events.values()
                     if _row['site_idx'] == site_idx
                ]

                # Get any new datasets that cluster with these (and aren't in a known site)
                known_site_events_new_sites = set(
                    [
                        new_site_idx
                        for event_id, new_site_idx
                        in event_sites.items()
                        if event_id in known_site_events
                    ]
                )
                new_overlapping_events = [
                    new_event_id
                    for new_event_id, new_site_idx
                    in event_sites.items()
                    if (new_site_idx in known_site_events_new_sites) & (new_event_id not in existing_events) & (new_event_id not in allocated_events)
                ]

                sites[site_idx] = Site(
                    known_site_events + new_overlapping_events,
                    site_info['centroid'],
                    site_info['Name'],
                    site_info['Comment']
                )
                # Allocate new events that have been used
                for _event_id in new_overlapping_events:
                    allocated_events.append(_event_id)

        if self.debug:
            print('event_id_array')
            print(event_id_array)
            print('clusters')
            print(clusters)

        # Then add new sites for any events that haven't already been explained.
        # Construct the sites
        for cluster in np.unique(clusters):
            cluster_event_id_array = event_id_array[clusters == cluster]

            # Get new, unallocated events
            new_site_events = [
                (str(event_id[0]), int(event_id[1]))
                for event_id
                in cluster_event_id_array
                if (str(event_id[0]), int(event_id[1])) not in allocated_events
            ]

            if len(new_site_events) != 0:

                sites[len(sites)+1] = Site(
                    new_site_events,
                    # np.mean(
                    #     centroid_array[clusters==cluster, :],
                    #     axis=0,
                    # ).flatten(),
                    np.array([0.0,0.0,0.0])
                )

        # plt.figure()
        # dn = scipy.cluster.hierarchy.dendrogram(linkage)
        # plt.savefig('dendrogram.png')

        return sites


# class HeirarchicalSiteModel:
#
#     def __init__(self, t=8.0):
#         self.t = t
#
#     def __call__(
#             self,
#             datasets: Dict[str, DatasetInterface],
#             events: Dict[Tuple[str, int], EventInterface],
#             processor: ProcessorInterface,
#             structure_array_refs,
#     ):
#
#         #
#         if len(events) == 0:
#             return {}
#
#         if len(events) == 1:
#             return {0: Site(
#                 [list(events.keys())[0], ],
#                 list(events.values())[0].centroid
#                 # np.mean(list(events.values())[0].pos_array, axis=1)
#             )}
#
#         # Choose a reference structure from highest res dataset
#         reference_dtag = min(datasets, key=lambda _dtag: datasets[_dtag].reflections.resolution())
#
#         # Get alignments to this reference
#         alignments: Dict[str, AlignmentInterface] = processor.process_dict(
#             {_dtag: Partial(Alignment.from_structure_arrays).paramaterise(
#                 _dtag,
#                 structure_array_refs[_dtag],
#                 structure_array_refs[reference_dtag],
#             ) for _dtag in datasets}
#         )
#
#         # Get the aligned centroids
#         aligned_centroids = []
#         for event_id, event in events.items():
#             _dtag, _event_idx = event_id
#             # unaligned_centroid = np.mean(event.pos_array, axis=0)
#             unaligned_centroid = event.centroid
#             alignment = alignments[_dtag]
#             transforms, com_ref, com_mov = alignment.get_transforms()
#             transform, associated_com_ref, closest_com_mov = _get_closest_transform(
#                 unaligned_centroid,
#                 transforms,
#                 com_ref,
#                 com_mov,
#             )
#             transformed_pos = transform.apply(gemmi.Position(*(unaligned_centroid - closest_com_mov)))
#             transformed_pos_np = np.array(
#                 [transformed_pos.x, transformed_pos.y, transformed_pos.z]) + associated_com_ref
#             aligned_centroids.append(transformed_pos_np.flatten())
#
#         # Get the array of centroids
#         centroid_array = np.array(
#             aligned_centroids
#         )
#
#         #
#         clusters = fclusterdata(
#             centroid_array,
#             t=self.t,
#             criterion="distance",
#             method="centroid"
#         )
#
#         #
#         unique_clusters = np.unique(clusters, )
#
#         # Merge the events based on the clustering
#         event_array = np.array([event_id for event_id in events.keys()])
#         j = 0
#         sites = {}
#         for cluster_num in unique_clusters:
#             site_event_ids = event_array[clusters == cluster_num]
#             # print([event_id for event_id in site_event_ids])
#             site_positions = np.concatenate(
#                 [events[(str(event_id[0]), int(event_id[1]))].pos_array for event_id in site_event_ids], axis=0)
#
#             sites[j] = Site(
#                 [(str(event_id[0]), int(event_id[1])) for event_id in site_event_ids],
#                 np.mean(site_positions, axis=0),
#             )
#             j += 1
#
#         return sites


class ResidueSiteModel:

    def get_event_environment(
            self,
            query_pos_array,
            structure_array,
            ns
    ):
        indexes = ns.query_ball_point(
            query_pos_array,
            5.0,
        )

        chains = structure_array.chains[indexes]
        residues = structure_array.seq_ids[indexes]

        res_ids = {}
        for chain, res in zip(chains, residues):
            res_ids[(str(chain), str(res))] = True

        return [res_id for res_id in res_ids.keys()]


    def get_event_environments(self, datasets, events):
        # Get the environments for each event
        event_evenvironments = {}
        for dtag, dataset in datasets.items():
            st_arr = StructureArray.from_structure(dataset.structure)
            ns = spatial.KDTree(st_arr.positions)
            dtag_events = {event_id: event for event_id, event in events.items() if event_id[0] == dtag}


            for event_id, event in dtag_events.items():
                event_evenvironments[event_id] = self.get_event_environment(
                    event.pos_array,
                    st_arr,
                    ns,
                )

        return event_evenvironments

    def get_overlap_graph(self, event_environments, min_overlap):
        g = nx.Graph()

        # Add the nodes
        for event_id in event_environments:
            g.add_node(event_id)

        # Form the site overlap matrix
        arr = np.zeros(
            (
                len(event_environments),
                len(event_environments),
            )
        )

        for j, event_1_id in enumerate(event_environments):
            environment_1 = event_environments[event_1_id]
            for k, event_2_id in enumerate(event_environments):
                environment_2 = event_environments[event_2_id]
                if event_1_id == event_2_id:
                    continue
                v = set(environment_1).intersection(set(environment_2))
                if len(v) > min_overlap:
                    arr[j, k] = 1

        # Complete the graph
        for idx, conn in np.ndenumerate(arr):
            x, y = idx

            if x == y:
                continue
            if conn:
                g.add_edge(x, y)

        return g


    def get_centroids(self, cliques, event_environments, reference ):
        centroids = {}
        for j, clique in enumerate(cliques):
            res_ids = set([res_id for _event_id in clique for res_id in event_environments[_event_id]])
            poss = [
                [atom.pos.x, atom.pos.y, atom.pos.z]
                for model in reference.structure.structure
                for chain in model
                for res in chain
                for atom in res
                if (chain.name, res.seqid.num) in res_ids
            ]
            if len(poss) == 0:
                centroid = [0.0,0.0,0.0]
            else:
                centroid = np.mean(
                    poss,
                    axis=1
                )
            centroids[j] = centroid

        return centroids

    def get_sites(self, event_environments, reference):
        # Get the overlap graph
        overlap_graph = self.get_overlap_graph(
            event_environments,
            3
        )

        # Get the disconnected Cliques
        cliques = list(nx.connected_components(overlap_graph))

        # Find the site centroids against the reference
        centroids = self.get_centroids(
            cliques,
            event_environments,
            reference,
        )

        # Construct the sites
        sites = {}
        for j, clique in enumerate(cliques):
            sites[j] = Site(
                [(str(event_id[0]), int(event_id[1])) for event_id in clique],
                centroids[j],
            )

        return sites




    def __call__(
            self,
            datasets: Dict[str, DatasetInterface],
            events: Dict[Tuple[str, int], EventInterface],
            ref_dataset
                 ):
        # Find the residue environment of each event (chain and residue number)
        event_environments: Dict[Tuple[str, int], List[Tuple[str, str]]] = self.get_event_environments(datasets, events)

        # Get the sites based on cliques of overlapping residues
        sites: Dict[int, Site] = self.get_sites(event_environments, ref_dataset)

        return sites

        ...
