"""Track-level coverage over explicitly annotated frames, never inferred labels."""


def summarize_continuity(tracks: dict[str, list[tuple[int, bool]]]) -> dict:
    results = {}
    for track_id, observations in sorted(tracks.items()):
        longest = run = interruptions = 0
        previous_index, previous_covered = -2, False
        ordered = sorted(observations)
        for index, covered in ordered:
            adjacent = index == previous_index + 1
            run = (run + 1 if adjacent else 1) if not covered else 0
            longest = max(longest, run)
            if adjacent and previous_covered and not covered:
                interruptions += 1
            previous_index, previous_covered = index, covered
        missed = sum(not covered for _, covered in ordered)
        results[track_id] = {
            "reviewed_visible_frames": len(ordered),
            "uncovered_frames": missed,
            "longest_uncovered_run_frames": longest,
            "covered_to_uncovered_transitions": interruptions,
            "protected_in_all_reviewed_frames": missed == 0,
        }
    return {
        "scope": "Only annotated frames; unreviewed gaps break runs. Not full-video protection.",
        "track_count": len(results),
        "fully_protected_reviewed_tracks": sum(
            r["protected_in_all_reviewed_frames"] for r in results.values()
        ),
        "tracks": results,
    }
