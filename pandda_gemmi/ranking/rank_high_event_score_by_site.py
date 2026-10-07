from ..interfaces import *

# def get_event_score(event):
#     return (event.build.signal / event.build.noise) * event.local_strength

def get_event_score(event):
    return event.score

class RankHighEventScoreBySite:
    def __call__(
            self,
            events: Dict[Tuple[str, int], EventInterface],
            sites: Dict[int, SiteInterface],
            autobuilds: Dict[Tuple[str, int], Dict[str, AutobuildInterface]],
            existing_sites,
    ):
        # # Sort sites ids by best score
        # site_scores = {}
        # for site_id, site in sites.items():
        #     site_scores[site_id] = max(
        #         [
        #             get_event_score(events[_event_id])
        #             for _event_id
        #             in site.event_ids
        #         ]
        #     )
        # # Sort sites ids numerically
        # site_scores = {}
        # for site_id, site in sites.items():
        #     site_scores[site_id] = max(
        #         [
        #             get_event_score(events[_event_id])
        #             for _event_id
        #             in site.event_ids
        #         ]
        #     )

        # Renumber sites by the highest event id in them IF FIRST RUN!
        if not existing_sites:
            sorted_sites = {}
            for j, site in enumerate(

                sorted(
                    sites.values(), 
                    # The invariant asserted here ("no event should have zero events")
                    # does not hold for ResiduePainting, which can emit a site with an
                    # empty event_ids -> "max() arg is an empty sequence". default keeps
                    # such a site, ranked last, rather than dropping it silently -- but
                    # why an empty site is produced belongs in the site model.
                    key=lambda _site: max(
                        [get_event_score(events[_event_id]) for _event_id in _site.event_ids],
                        default=float("-inf"),
                    ),
                    reverse=True
                ),
            ):
                sorted_sites[j+1] = site
        else:
            for _site_id in existing_sites:
                sorted_sites[_site_id] = sites[_site_id]

            for j, site in enumerate(

                sorted(
                    [_site for _site_id, _site in sites if _site_id not in sorted_sites], 
                    key=lambda _site: max([get_event_score(events[_event_id]) for _event_id in _site.event_ids]),  # Max is safe: no event should have zero events 
                    reverse=True
                ),
            ):
                sorted_sites[j+1] = site
        # Sort event ids within each site by best score
        sorted_event_ids = []
        for _site_id in sorted(sorted_sites):
            for event_id in sorted(
                    sorted_sites[_site_id].event_ids,
                    key=lambda _event_id: get_event_score(events[_event_id]),
                    reverse=True,
            ):
                sorted_event_ids.append(event_id)

        return sorted_event_ids, sorted_sites
